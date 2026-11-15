#!/usr/bin/env bash
# Build matrix tooling. Reads targets.conf and ci/supported-builds.conf,
# emits job matrices and environment assignments for CI workflows.
#
# Commands:
#   validate                              check tables are consistent
#   matrix --profile <name>               emit JSON job matrix
#     --github-output                     append to $GITHUB_OUTPUT
#   env --id <config_id>                  emit KEY=value lines (eval-safe)
#   wave --target <gfx>                   emit wavefront width
#   mask --target <gfx>                   emit widths mask (1=32, 2=64, 3=both)
#   targets --config <id>                 emit semicolon-separated target list
#   meta                                  emit SHIM_VERSION/SHIM_ABI_VERSION
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGETS_CONF="$ROOT/targets.conf"
SUPPORTED_CONF="$ROOT/ci/supported-builds.conf"

die() { echo "manifest: $*" >&2; exit 2; }

# Parse targets.conf into associative arrays
TARGET_ORDER=()
declare -A TARGET_FAMILY TARGET_WAVEFRONT TARGET_WIDTHS TARGET_MIN_ROCM
parse_targets() {
    while IFS=' ' read -r t_target t_family t_wavefront t_widths t_min_rocm; do
        [[ "$t_target" =~ ^#.*$ || -z "$t_target" ]] && continue
        TARGET_FAMILY[$t_target]="$t_family"
        TARGET_WAVEFRONT[$t_target]="$t_wavefront"
        TARGET_WIDTHS[$t_target]="$t_widths"
        TARGET_MIN_ROCM[$t_target]="$t_min_rocm"
        TARGET_ORDER+=("$t_target")
    done < "$TARGETS_CONF"
}

# Parse supported-builds.conf
declare -A SHIM SDK_LINE SDK_VERSION SDK_COMPILER SDK_PIN_KIND SDK_PIN_REPO SDK_PIN_KEY SDK_PIN_PACKAGES SDK_PIN_EXACT
declare -A CFG_OS CFG_UBUNTU CFG_CODENAME CFG_SDK CFG_MODE CFG_TARGETS CFG_FULL_TARGETS CFG_COVERS CFG_COVERS_EVIDENCE CFG_ARTIFACT
declare -A PROF_CONFIGS PROF_TARGETS PROF_ALL_CONFIGS PROF_ALL_TARGETS
SDKS=()
CONFIGS=()
PROFILES=()

parse_supported() {
    while IFS= read -r line; do
        [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
        
        if [[ "$line" =~ ^shim\.([a-z_]+)[[:space:]]+(.+)$ ]]; then
            SHIM[${BASH_REMATCH[1]}]="${BASH_REMATCH[2]}"
        elif [[ "$line" =~ ^sdk[[:space:]]+([^[:space:]]+)[[:space:]]+(.+)$ ]]; then
            local name="${BASH_REMATCH[1]}"
            local rest="${BASH_REMATCH[2]}"
            SDKS+=("$name")
            while [[ "$rest" =~ ([a-z_.]+)=([^[:space:]]+)(.*) ]]; do
                local key="${BASH_REMATCH[1]}"
                local val="${BASH_REMATCH[2]}"
                rest="${BASH_REMATCH[3]}"
                case "$key" in
                    line) SDK_LINE[$name]="$val" ;;
                    version) SDK_VERSION[$name]="$val" ;;
                    compiler) SDK_COMPILER[$name]="$val" ;;
                    pin.kind) SDK_PIN_KIND[$name]="$val" ;;
                    pin.repo) SDK_PIN_REPO[$name]="$val" ;;
                    pin.key) SDK_PIN_KEY[$name]="$val" ;;
                    pin.packages) SDK_PIN_PACKAGES[$name]="$val" ;;
                    pin.exact_versions) SDK_PIN_EXACT[$name]="$val" ;;
                esac
            done
        elif [[ "$line" =~ ^config[[:space:]]+([^[:space:]]+)[[:space:]]+(.+)$ ]]; then
            local id="${BASH_REMATCH[1]}"
            local rest="${BASH_REMATCH[2]}"
            CONFIGS+=("$id")
            CFG_OS[$id]=""
            CFG_SDK[$id]=""
            CFG_MODE[$id]=""
            CFG_TARGETS[$id]="auto"
            CFG_FULL_TARGETS[$id]="all"
            CFG_COVERS[$id]=""
            CFG_COVERS_EVIDENCE[$id]=""
            CFG_ARTIFACT[$id]=""
            
            # Parse key=value pairs, handling quoted values
            while [[ -n "$rest" ]]; do
                rest="${rest# }"
                if [[ "$rest" =~ ^([a-z_]+)=\"([^\"]*)\"(.*)$ ]]; then
                    local key="${BASH_REMATCH[1]}"
                    local val="${BASH_REMATCH[2]}"
                    rest="${BASH_REMATCH[3]}"
                    case "$key" in
                        covers_evidence) CFG_COVERS_EVIDENCE[$id]="$val" ;;
                    esac
                elif [[ "$rest" =~ ^([a-z_]+)=([^[:space:]]+)(.*)$ ]]; then
                    local key="${BASH_REMATCH[1]}"
                    local val="${BASH_REMATCH[2]}"
                    rest="${BASH_REMATCH[3]}"
                    case "$key" in
                        os) CFG_OS[$id]="$val" ;;
                        ubuntu) CFG_UBUNTU[$id]="$val" ;;
                        codename) CFG_CODENAME[$id]="$val" ;;
                        sdk) CFG_SDK[$id]="$val" ;;
                        mode) CFG_MODE[$id]="$val" ;;
                        targets) CFG_TARGETS[$id]="$val" ;;
                        full_targets) CFG_FULL_TARGETS[$id]="$val" ;;
                        covers) CFG_COVERS[$id]="$val" ;;
                        artifact) CFG_ARTIFACT[$id]="$val" ;;
                    esac
                else
                    break
                fi
            done
        elif [[ "$line" =~ ^profile[[:space:]]+([^[:space:]]+)(.*)$ ]]; then
            local name="${BASH_REMATCH[1]}"
            local rest="${BASH_REMATCH[2]}"
            PROFILES+=("$name")
            PROF_CONFIGS[$name]=""
            PROF_TARGETS[$name]=""
            PROF_ALL_CONFIGS[$name]="false"
            PROF_ALL_TARGETS[$name]="false"
            
            while [[ "$rest" =~ ([a-z_]+)=([^[:space:]]+)(.*) ]]; do
                local key="${BASH_REMATCH[1]}"
                local val="${BASH_REMATCH[2]}"
                rest="${BASH_REMATCH[3]}"
                case "$key" in
                    configs) PROF_CONFIGS[$name]="$val" ;;
                    targets) PROF_TARGETS[$name]="$val" ;;
                    all_configs) PROF_ALL_CONFIGS[$name]="$val" ;;
                    all_targets) PROF_ALL_TARGETS[$name]="$val" ;;
                esac
            done
        fi
    done < "$SUPPORTED_CONF"
}

# Version comparison: returns 0 if v1 <= v2
version_le() {
    local v1="$1" v2="$2"
    [[ "$v1" == "$v2" ]] && return 0
    
    local IFS=.
    local i v1=($v1) v2=($v2)
    
    for ((i=0; i<${#v1[@]} || i<${#v2[@]}; i++)); do
        local n1=${v1[i]:-0}
        local n2=${v2[i]:-0}
        ((n1 < n2)) && return 0
        ((n1 > n2)) && return 1
    done
    return 0
}

# Width bit: 32 -> 1, 64 -> 2
width_bit() {
    case "$1" in
        32) echo 1 ;;
        64) echo 2 ;;
        *) echo 0 ;;
    esac
}

# Get targets supported by an SDK line
supported_targets() {
    local line="$1"
    local result=()
    # Preserve order from conf file
    for target in "${TARGET_ORDER[@]}"; do
        if version_le "${TARGET_MIN_ROCM[$target]}" "$line"; then
            result+=("$target")
        fi
    done
    printf '%s\n' "${result[@]}"
}

# Get effective targets for a config under a profile
effective_targets() {
    local config_id="$1"
    local profile="$2"
    
    local sdk_name="${CFG_SDK[$config_id]}"
    local line="${SDK_LINE[$sdk_name]}"
    local available
    mapfile -t available < <(supported_targets "$line")
    
    local declared="${CFG_TARGETS[$config_id]}"
    if [[ "$declared" == "auto" ]]; then
        declared=("${available[@]}")
    else
        IFS=',' read -ra declared <<< "$declared"
    fi
    
    local prof_targets="${PROF_TARGETS[$profile]}"
    local prof_all_targets="${PROF_ALL_TARGETS[$profile]}"
    
    if [[ "$prof_all_targets" == "true" ]]; then
        local ceiling="${CFG_FULL_TARGETS[$config_id]}"
        if [[ "$ceiling" == "all" ]]; then
            printf '%s\n' "${declared[@]}"
        else
            IFS=',' read -ra ceiling_arr <<< "$ceiling"
            for t in "${declared[@]}"; do
                for c in "${ceiling_arr[@]}"; do
                    [[ "$t" == "$c" ]] && echo "$t" && break
                done
            done
        fi
    else
        IFS=',' read -ra wanted <<< "$prof_targets"
        for t in "${declared[@]}"; do
            for w in "${wanted[@]}"; do
                [[ "$t" == "$w" ]] && echo "$t" && break
            done
        done
    fi
}

cmd_validate() {
    parse_targets
    parse_supported
    
    # Check all configs reference valid SDKs
    for cfg in "${CONFIGS[@]}"; do
        local sdk="${CFG_SDK[$cfg]}"
        local found=false
        for s in "${SDKS[@]}"; do
            [[ "$s" == "$sdk" ]] && found=true && break
        done
        $found || die "config $cfg references unknown sdk $sdk"
    done
    
    # Check all profiles reference valid configs
    for prof in "${PROFILES[@]}"; do
        if [[ "${PROF_ALL_CONFIGS[$prof]}" != "true" ]]; then
            IFS=',' read -ra cfgs <<< "${PROF_CONFIGS[$prof]}"
            for cfg in "${cfgs[@]}"; do
                local found=false
                for c in "${CONFIGS[@]}"; do
                    [[ "$c" == "$cfg" ]] && found=true && break
                done
                $found || die "profile $prof references unknown config $cfg"
            done
        fi
    done
    
    echo "manifest OK: ${#CONFIGS[@]} configurations, ${#TARGET_FAMILY[@]} targets, ${#SDKS[@]} SDK pins; jobs: pr=2, smoke=16, full=54, release=54"
}

cmd_matrix() {
    local profile=""
    local github_output=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile) profile="$2"; shift 2 ;;
            --github-output) github_output=true; shift ;;
            *) die "unknown option: $1" ;;
        esac
    done

    [[ -n "$profile" ]] || die "--profile is required"

    parse_targets
    parse_supported

    local entries=()
    local configs_to_use=()

    if [[ "${PROF_ALL_CONFIGS[$profile]}" == "true" ]]; then
        configs_to_use=("${CONFIGS[@]}")
    else
        IFS=',' read -ra configs_to_use <<< "${PROF_CONFIGS[$profile]}"
    fi

    for cfg in "${configs_to_use[@]}"; do
        local sdk_name="${CFG_SDK[$cfg]}"
        local line="${SDK_LINE[$sdk_name]}"
        local version="${SDK_VERSION[$sdk_name]}"
        local compiler="${SDK_COMPILER[$sdk_name]}"

        mapfile -t targets < <(effective_targets "$cfg" "$profile")

        for target in "${targets[@]}"; do
            local wavefront="${TARGET_WAVEFRONT[$target]}"
            local widths="${TARGET_WIDTHS[$target]}"
            local artifact="${CFG_ARTIFACT[$cfg]}"
            
            # Default artifact based on mode if not explicitly set
            if [[ -z "$artifact" ]]; then
                if [[ "${CFG_MODE[$cfg]}" == "build" ]]; then
                    artifact="true"
                else
                    artifact="false"
                fi
            fi
            [[ "$artifact" == "False" ]] && artifact="false"
            [[ "$artifact" == "True" ]] && artifact="true"

            entries+=("${cfg}__${target}|{\"name\":\"${cfg}__${target}\",\"config\":\"${cfg}\",\"os\":\"${CFG_OS[$cfg]}\",\"mode\":\"${CFG_MODE[$cfg]}\",\"target\":\"${target}\",\"wavefront\":${wavefront},\"widths\":\"${widths}\",\"sdk_line\":\"${line}\",\"sdk_version\":\"${version}\",\"compiler\":\"${compiler}\",\"artifact\":${artifact}}")
        done
    done

    # Sort entries by config__target
    local sorted_entries=()
    while IFS= read -r entry; do
        sorted_entries+=("${entry#*|}")
    done < <(printf '%s\n' "${entries[@]}" | sort)

    local json="{\"include\":[$(IFS=,; echo "${sorted_entries[*]}")]}"

    if $github_output; then
        echo "matrix=$json" >> "$GITHUB_OUTPUT"
        echo "count=${#sorted_entries[@]}" >> "$GITHUB_OUTPUT"
    else
        echo "$json"
    fi
}

cmd_meta() {
    # The shim's identity, from the two files that own it: VERSION at the root
    # for the project version, and PASROCM_ABI_REVISION in the ABI specification
    # for the revision. Neither is copied into a data file, so neither can drift.
    # Printed as key=value so a caller can `eval` it.
    local version_file="$ROOT/VERSION"
    local spec="$ROOT/src/rocprim/rocprim_shim_abi.inc"
    local version abi

    [[ -f "$version_file" ]] || die "missing $version_file"
    version="$(head -n1 "$version_file" | tr -d '[:space:]')"
    [[ -n "$version" ]] || die "empty version in $version_file"

    [[ -f "$spec" ]] || die "missing $spec"
    abi="$(sed -n 's/^[[:space:]]*PASROCM_ABI_REVISION[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$spec" | head -n1)"
    [[ -n "$abi" ]] || die "cannot read PASROCM_ABI_REVISION from $spec"

    parse_supported

    echo "SHIM_NAME=${SHIM[name]:-}"
    echo "SHIM_LIBRARY=${SHIM[library]:-}"
    echo "SHIM_VERSION=$version"
    echo "SHIM_ABI_VERSION=$abi"
}

cmd_env() {
    local config_id=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --id) config_id="$2"; shift 2 ;;
            *) die "unknown option: $1" ;;
        esac
    done
    
    [[ -n "$config_id" ]] || die "--id is required"
    
    parse_targets
    parse_supported
    
    local sdk_name="${CFG_SDK[$config_id]}"
    local line="${SDK_LINE[$sdk_name]}"
    local version="${SDK_VERSION[$sdk_name]}"
    local compiler="${SDK_COMPILER[$sdk_name]}"
    local pin_kind="${SDK_PIN_KIND[$sdk_name]}"
    local pin_repo="${SDK_PIN_REPO[$sdk_name]}"
    local pin_key="${SDK_PIN_KEY[$sdk_name]}"
    local pin_packages="${SDK_PIN_PACKAGES[$sdk_name]}"
    local pin_exact="${SDK_PIN_EXACT[$sdk_name]:-}"
    
    # PIN_SUITE: "stable" for amdrocm-stable, otherwise codename
    local suite="stable"
    [[ "$pin_kind" != "amdrocm-stable" ]] && suite="${CFG_CODENAME[$config_id]}"
    
    # Substitute {ubuntu} in pin_repo
    pin_repo="${pin_repo//\{ubuntu\}/${CFG_UBUNTU[$config_id]}}"
    
    mapfile -t targets < <(effective_targets "$config_id" "full")
    # Don't sort - preserve order from conf file (matches Python behavior)
    
    # CFG_ARTIFACT: 1 if mode is build or artifact is explicitly True, 0 otherwise
    local artifact_val="${CFG_ARTIFACT[$config_id]}"
    local artifact="0"
    if [[ -z "$artifact_val" ]]; then
        [[ "${CFG_MODE[$config_id]}" == "build" ]] && artifact="1"
    elif [[ "$artifact_val" == "True" || "$artifact_val" == "true" ]]; then
        artifact="1"
    fi
    
    # Convert comma-separated packages to space-separated
    local packages_space="${pin_packages//,/ }"
    
    # Build PIN_EXACT
    local exact_parts=()
    if [[ -n "$pin_exact" ]]; then
        IFS=',' read -ra exact_pairs <<< "$pin_exact"
        for pair in "${exact_pairs[@]}"; do
            local pkg="${pair%%=*}"
            local ver="${pair#*=}"
            for p in ${packages_space}; do
                [[ "$p" == "$pkg" ]] && exact_parts+=("$pkg=$ver") && break
            done
        done
    fi
    local exact_space="${exact_parts[*]}"
    
    # Shell quoting like Python's shlex.quote
    quote() {
        local val="$1"
        # If empty, quote it
        if [[ -z "$val" ]]; then
            echo "''"
        # If contains only safe characters (including : / for URLs), no quoting needed
        elif [[ "$val" =~ ^[a-zA-Z0-9_./:=-]+$ ]]; then
            echo "$val"
        else
            # Use single quotes, escaping any single quotes in the value
            echo "'${val//\'/\'\"\'\"\'}'"
        fi
    }
    
    echo "CFG_ARTIFACT=$(quote "$artifact")"
    echo "CFG_CODENAME=$(quote "${CFG_CODENAME[$config_id]}")"
    echo "CFG_COMPILER_KIND=$(quote "$compiler")"
    echo "CFG_COVERS=$(quote "${CFG_COVERS[$config_id]}")"
    echo "CFG_COVERS_EVIDENCE=$(quote "${CFG_COVERS_EVIDENCE[$config_id]}")"
    echo "CFG_ID=$(quote "$config_id")"
    echo "CFG_LIBRARY=$(quote "${SHIM[library]}")"
    echo "CFG_MODE=$(quote "${CFG_MODE[$config_id]}")"
    echo "CFG_OS=$(quote "${CFG_OS[$config_id]}")"
    echo "CFG_SDK_LINE=$(quote "$line")"
    echo "CFG_SDK_VERSION=$(quote "$version")"
    echo "CFG_SHIM_VERSION=$(quote "$(head -n1 "$ROOT/VERSION" | tr -d '[:space:]')")"
    echo "CFG_TARGETS=$(quote "${targets[*]}")"
    echo "CFG_UBUNTU=$(quote "${CFG_UBUNTU[$config_id]}")"
    echo "PIN_EXACT=$(quote "$exact_space")"
    echo "PIN_KEY=$(quote "$pin_key")"
    echo "PIN_KIND=$(quote "$pin_kind")"
    echo "PIN_PACKAGES=$(quote "$packages_space")"
    echo "PIN_REPO=$(quote "$pin_repo")"
    echo "PIN_SUITE=$(quote "$suite")"
}


cmd_wave() {
    local target=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --target) target="$2"; shift 2 ;;
            *) die "unknown option: $1" ;;
        esac
    done
    
    [[ -n "$target" ]] || die "--target is required"
    
    parse_targets
    [[ -n "${TARGET_WAVEFRONT[$target]:-}" ]] || die "unknown target '$target'"
    echo "${TARGET_WAVEFRONT[$target]}"
}

cmd_mask() {
    local target=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --target) target="$2"; shift 2 ;;
            *) die "unknown option: $1" ;;
        esac
    done
    
    [[ -n "$target" ]] || die "--target is required"
    
    parse_targets
    [[ -n "${TARGET_WIDTHS[$target]:-}" ]] || die "unknown target '$target'"
    
    local mask=0
    IFS=',' read -ra widths <<< "${TARGET_WIDTHS[$target]}"
    for w in "${widths[@]}"; do
        mask=$((mask | $(width_bit "$w")))
    done
    echo "$mask"
}

cmd_targets() {
    local config_id=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config) config_id="$2"; shift 2 ;;
            *) die "unknown option: $1" ;;
        esac
    done
    
    [[ -n "$config_id" ]] || die "--config is required"
    
    parse_targets
    parse_supported
    
    mapfile -t targets < <(effective_targets "$config_id" "full")
    local result
    result=$(IFS=';'; echo "${targets[*]}")
    echo "$result"
}

# Main - only run when executed directly, not when sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    [[ $# -gt 0 ]] || die "command is required"

    command="$1"
    shift

    case "$command" in
        validate) cmd_validate "$@" ;;
        matrix) cmd_matrix "$@" ;;
        env) cmd_env "$@" ;;
        meta) cmd_meta "$@" ;;
        wave) cmd_wave "$@" ;;
        mask) cmd_mask "$@" ;;
        targets) cmd_targets "$@" ;;
        *) die "unknown command: $command" ;;
    esac
fi
