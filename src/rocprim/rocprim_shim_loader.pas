{------------------------------------------------------------------------------
  PasROCm - Free Pascal bindings for the AMD ROCm libraries.

  Copyright (c) 2024-2026 Eudox67

  MIT License

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.

  This is an unofficial project and is not affiliated with or endorsed by AMD.
------------------------------------------------------------------------------}

{ Runtime loader for the rocPRIM shim.

  The shim is built once per GPU target (libpasrocm_shim_<gfx>.so); this unit
  picks the file that matches the device and hands the entry points to
  rocprim_bindings. The order is fixed and deliberately paranoid:

    1. detect the device's gfx target with hipGetDeviceProperties;
    2. find and dlopen the matching library;
    3. read pasrocm_shim_target / _wavefront_size / _rocm_version /
       _wavefront_widths_mask through the pure exports -- no kernels yet;
    4. refuse the library if target or wavefront disagree with the device;
    5. only then resolve the 144 rocPRIM entry points, and only if all of them
       resolve (the table is never left half-bound).

  A wrong library is therefore a named exception with the two targets in it,
  not a mis-sized kernel or a call through a nil pointer.

  Load once, lazily: PasROCmShimEnsureLoaded is called by every wrapper in
  rocprim_bindings, so a program that never touches rocPRIM never opens it.
}

unit rocprim_shim_loader;

{$mode objfpc}{$H+}

interface

uses
  SysUtils;

type
  { Base class for every loader failure. Catching this catches all of them. }
  EShimError = class(Exception);

  { No candidate file existed, or PASROCM_SHIM points at nothing. }
  EShimLibraryNotFound = class(EShimError);

  { dlopen failed, or no usable HIP device could be queried. }
  EShimLoadFailed = class(EShimError);

  { The library opened but is not the right one: wrong GPU target, a width the
    hardware cannot run, or an SDK line the caller explicitly excluded. }
  EShimIdentityError = class(EShimError);

  { The library opened and identified itself, but does not export the whole
    table -- typically an older or truncated build. }
  EShimBindError = class(EShimError);

  { Outcome of comparing a loaded shim with the detected device. Pure decision,
    so the policy is testable without a GPU (tests/Test_pasrocm_shim_loader). }
  TShimIdentityVerdict = (
    ivOk,
    ivTargetMismatch,
    ivBadCompiledWidth,
    ivBadMask,
    ivWidthMismatch
  );

  TShimPathArray = array of string;

  TShimLoadInfo = record
    { File that was opened. }
    LibraryPath: string;
    { What the library says it was built for (pasrocm_shim_target). }
    Target: string;
    { SDK line the library was built against (pasrocm_shim_rocm_version). }
    RocmVersion: string;
    WavefrontSize: Cardinal;
    WidthsMask: Cardinal;
    { ABI revision the library reports (pasrocm_shim_abi_version). }
    AbiVersion: Cardinal;
    { Deterministic provenance string (pasrocm_shim_build_info). }
    BuildInfo: string;
    { GPU the loader selected (normalized gcnArchName). }
    DeviceGfx: string;
    DeviceIndex: Integer;
    DeviceWarpSize: Integer;
    { hipRuntimeGetVersion, or -1 when it could not be read. }
    HipRuntimeVersion: Int32;
    { True when PASROCM_EXPECT_GFX supplied DeviceGfx instead of the device. }
    DeviceOverridden: Boolean;
    { Non-fatal findings (SDK-line mismatch, unpinned provenance), ';'-joined. }
    Warnings: string;
  end;

const
  { Exact file to open. Authoritative: if set, no other candidate is tried,
    and a missing file is an error rather than a silent fallback. }
  SHIM_ENV_FILE = 'PASROCM_SHIM';
  { Directory holding libpasrocm_shim_<gfx>.so (the LD_LIBRARY_PATH-free way). }
  SHIM_ENV_DIR = 'PASROCM_SHIM_DIR';
  { Device index to detect (default 0). }
  SHIM_ENV_DEVICE = 'PASROCM_DEVICE';
  { Trust this gfx instead of querying HIP (CI, host-only identity checks). }
  SHIM_ENV_EXPECT_GFX = 'PASROCM_EXPECT_GFX';
  { Require the shim's SDK stamp to start with this prefix. }
  SHIM_ENV_EXPECT_ROCM = 'PASROCM_EXPECT_ROCM';
  { Upgrade the SDK-line warning to a fatal identity error. }
  SHIM_ENV_STRICT_ROCM = 'PASROCM_STRICT_ROCM';

{ 'gfx90a:sramecc+:xnack-' -> 'gfx90a'. Lowercase, trimmed, feature suffixes
  dropped: the file name carries the architecture, not the feature set. }
function PasROCmNormalizeGfx(const AName: string): string;

{ libpasrocm_shim_<gfx>.so (Linux) / pasrocm_shim_<gfx>.dll (Windows). }
function PasROCmShimLibraryFileName(const AGfx: string): string;

{ Every path the loader will try, in order, ending with the bare file name so
  that LD_LIBRARY_PATH keeps working. PASROCM_SHIM, when set, comes first and
  is the only entry considered by the loader. Pure: reads no files. }
function PasROCmShimCandidatePaths(const AGfx: string): TShimPathArray;

{ Pure identity policy: target, compiled width, mask, and the shim's own
  pasrocm_wavefront_compatible() result (passed in, never re-derived). }
function PasROCmClassifyIdentity(const AShimTarget, ADeviceGfx: string;
  ACompiledWidth, AReportedWidth, AAllowedMask: Cardinal;
  ACompatOk: Boolean): TShimIdentityVerdict;

{ ABI gate. A revision of 0 means the library does not report one (a pre-1
  build); any difference means the prototypes this executable was generated
  from do not describe that library. Pure, so the rule is testable. }
function PasROCmAbiVersionAccepted(ALoaded, AExpected: Cardinal): Boolean;

{ Open the shim for this device if it is not open yet. Raises EShimError (one
  of the subclasses) on failure, and re-raises the latched error on later
  calls. Idempotent and safe from several threads. }
function PasROCmShimEnsureLoaded: Boolean;

{ Same as PasROCmShimEnsureLoaded, named for explicit start-up calls. }
function PasROCmShimLoad: Boolean;

{ Drop the library and clear every entry point. Tests and re-detection only:
  never unload while kernels may still be in flight on another thread. }
procedure PasROCmShimUnload;

function PasROCmShimLoaded: Boolean;
function PasROCmShimInfo: TShimLoadInfo;
function PasROCmShimLastError: string;

implementation

uses
  dynlibs, syncobjs, hip_types, hip_error, hip_runtime, rocprim_shim_api,
  rocprim_shim_hash;

const
  { Expected SHA256 of the library file; overrides the .sha256 sidecar.
    Deliberately NOT in the interface: adding a constant to this unit's
    interface makes FPC 3.3.1 abort with 'Internal error 200611011' while
    compiling rocprim_bindings.pas (measured 2026-11-15, bisected to the
    interface declaration alone). Keep new loader declarations here. }
  SHIM_ENV_SHA256 = 'PASROCM_SHIM_SHA256';

{$IFDEF UNIX}
{ Read the live environment. Measured on FPC 3.3.1: GetEnvironmentVariable
  snapshots the environment at start-up, so a host program (or a test) that
  sets PASROCM_SHIM_DIR at run time would be ignored even though the variable
  is set for every child process. getenv has no such cache. }
function libc_getenv(AName: PAnsiChar): PAnsiChar;
  cdecl; external 'c' name 'getenv';
{$ENDIF}

type
  TShimState = (ssIdle, ssLoaded, ssFailed);

var
  { Serializes the transition only; the steady state is a single test of GState. }
  GLock: TCriticalSection;
  GState: TShimState = ssIdle;
  GInfo: TShimLoadInfo;
  GError: string;
  GHandle: TLibHandle = NilHandle;

function EnvStr(const AName: string): string;
{$IFDEF UNIX}
var
  Value: PAnsiChar;
{$ENDIF}
begin
{$IFDEF UNIX}
  Value := libc_getenv(PAnsiChar(AnsiString(AName)));
  if Value = nil then
    Result := ''
  else
    Result := Trim(string(Value));
{$ELSE}
  Result := Trim(GetEnvironmentVariable(AName));
{$ENDIF}
end;

{ Leading integer of a version stamp: '10.1.0' -> 10, '7.16.26385-...' -> 7,
  'unknown' -> 0. Used only to compare SDK lines, never to compare builds. }
function VersionMajor(const AVersion: string): Integer;
var
  Index: Integer;
begin
  Result := 0;
  Index := 1;
  while (Index <= Length(AVersion)) and (AVersion[Index] >= '0')
    and (AVersion[Index] <= '9') do
  begin
    Result := Result * 10 + (Ord(AVersion[Index]) - Ord('0'));
    Inc(Index);
  end;
end;

function PasROCmNormalizeGfx(const AName: string): string;
var
  Text: string;
  Colon: SizeInt;
begin
  Text := Trim(AName);
  Colon := Pos(':', Text);
  if Colon > 0 then
    Text := Copy(Text, 1, Colon - 1);
  Result := LowerCase(Trim(Text));
end;

function PasROCmShimLibraryFileName(const AGfx: string): string;
begin
  {$IFDEF MSWINDOWS}
  Result := 'pasrocm_shim_' + AGfx + '.dll';
  {$ELSE}
  Result := 'libpasrocm_shim_' + AGfx + '.so';
  {$ENDIF}
end;

procedure AddCandidate(var AList: TShimPathArray; const APath: string);
var
  Index: Integer;
begin
  if APath = '' then
    Exit;
  for Index := 0 to High(AList) do
    if AList[Index] = APath then
      Exit;
  SetLength(AList, Length(AList) + 1);
  AList[High(AList)] := APath;
end;

function PasROCmShimCandidatePaths(const AGfx: string): TShimPathArray;
var
  FileName: string;
  ExeDir: string;
  HomeDir: string;

  procedure AddDir(const ADir: string);
  begin
    if ADir <> '' then
      AddCandidate(Result, IncludeTrailingPathDelimiter(ADir) + FileName);
  end;

begin
  Result := nil;
  FileName := PasROCmShimLibraryFileName(AGfx);
  ExeDir := ExtractFileDir(ExpandFileName(ParamStr(0)));
  HomeDir := GetEnvironmentVariable('HOME');

  { An explicit file outranks everything; the loader does not fall back. }
  AddCandidate(Result, EnvStr(SHIM_ENV_FILE));
  AddDir(EnvStr(SHIM_ENV_DIR));

  { Next to the executable, then the layout a package tarball unpacks to. }
  AddDir(ExeDir);
  AddDir(ExeDir + PathDelim + '..' + PathDelim + 'lib');
  AddDir(ExeDir + PathDelim + '..' + PathDelim + '..' + PathDelim + 'lib');

  { Per-user and system install locations. }
  if HomeDir <> '' then
    AddDir(HomeDir + PathDelim + '.local' + PathDelim + 'lib' + PathDelim + 'pasrocm');
  AddDir('/usr/local/lib/pasrocm');
  AddDir('/usr/lib/pasrocm');
  AddDir('/usr/lib64/pasrocm');

  { A checkout: ./src/rocprim, so test.sh and the examples need no env at all. }
  AddDir(GetCurrentDir + PathDelim + 'src' + PathDelim + 'rocprim');

  { Last resort: the bare name, resolved by the dynamic loader itself, which
    keeps LD_LIBRARY_PATH working as it did before this unit existed. }
  AddCandidate(Result, FileName);
end;

function PasROCmAbiVersionAccepted(ALoaded, AExpected: Cardinal): Boolean;
begin
  Result := (ALoaded <> 0) and (ALoaded = AExpected);
end;

function PasROCmClassifyIdentity(const AShimTarget, ADeviceGfx: string;
  ACompiledWidth, AReportedWidth, AAllowedMask: Cardinal;
  ACompatOk: Boolean): TShimIdentityVerdict;
var
  CompiledBit: Cardinal;
begin
  if not SameText(Trim(AShimTarget), Trim(ADeviceGfx)) then
    Exit(ivTargetMismatch);

  if ACompiledWidth = 32 then
    CompiledBit := 1
  else if ACompiledWidth = 64 then
    CompiledBit := 2
  else
    Exit(ivBadCompiledWidth);

  if (AAllowedMask = 0) or (AAllowedMask > 3) then
    Exit(ivBadMask);

  { A build for a width the hardware cannot run is an impossible library. }
  if (AAllowedMask and CompiledBit) = 0 then
    Exit(ivBadMask);

  if (AReportedWidth <> 32) and (AReportedWidth <> 64) then
    Exit(ivWidthMismatch);

  if not ACompatOk then
    Exit(ivWidthMismatch);

  Result := ivOk;
end;

{ Query the device's architecture. The device index is PASROCM_DEVICE or 0; we
  do not call hipGetDevice, because the property query takes the ordinal and
  works even when another device is current. }
function DetectDevice(out AGfx: string; out AWarpSize, AIndex: Int32;
  out AError: string): Boolean;
var
  Prop: THipDeviceProp;
  St: THipError;
  Text: string;
begin
  Result := False;
  AIndex := StrToIntDef(EnvStr(SHIM_ENV_DEVICE), 0);
  if AIndex < 0 then
    AIndex := 0;

  FillChar(Prop, SizeOf(Prop), 0);
  St := hipGetDeviceProperties(Prop, AIndex);
  if St <> hipSuccess then
  begin
    AError := Format('HIP device %d is not usable: %s (%s)',
      [AIndex, string(hipGetErrorName(St)), string(hipGetErrorString(St))]);
    Exit;
  end;

  Text := PAnsiChar(@Prop.gcnArchName[0]);
  AGfx := PasROCmNormalizeGfx(Text);
  AWarpSize := Prop.warpSize;
  if AGfx = '' then
  begin
    AError := Format('HIP device %d (%s) reports no gcnArchName',
      [AIndex, PAnsiChar(@Prop.name[0])]);
    Exit;
  end;
  Result := True;
end;

{ Resolve one table entry by name into its slot. Returns False and names the
  symbol when the library does not export it. }
function BindOne(AHandle: TLibHandle; AIndex: Integer;
  out AMissing: string): Boolean;
var
  SymbolName: PAnsiChar;
  Slot: PPointer;
  Address: Pointer;
begin
  Result := False;
  SymbolName := rocprim_shim_symbol_name(AIndex);
  Slot := rocprim_shim_symbol_slot(AIndex);
  if (SymbolName = nil) or (Slot = nil) then
  begin
    AMissing := Format('<table index %d>', [AIndex]);
    Exit;
  end;
  Address := GetProcedureAddress(AHandle, string(SymbolName));
  if Address = nil then
  begin
    AMissing := string(SymbolName);
    Exit;
  end;
  Slot^ := Address;
  Result := True;
end;

{ Resolve the whole table, but publish it only when nothing is missing: a
  half-bound table is worse than a refusal. }
procedure BindAll(AHandle: TLibHandle; out AMissing: TShimPathArray);
var
  Index: Integer;
  Address: Pointer;
  Resolved: array of Pointer;
begin
  AMissing := nil;
  SetLength(Resolved, ROCPRIM_SHIM_SYMBOL_COUNT);
  for Index := 0 to ROCPRIM_SHIM_SYMBOL_COUNT - 1 do
  begin
    Address := GetProcedureAddress(AHandle,
      string(rocprim_shim_symbol_name(Index)));
    if Address = nil then
    begin
      SetLength(AMissing, Length(AMissing) + 1);
      AMissing[High(AMissing)] := string(rocprim_shim_symbol_name(Index));
    end
    else
      Resolved[Index] := Address;
  end;
  if Length(AMissing) = 0 then
    for Index := 0 to High(Resolved) do
      rocprim_shim_symbol_slot(Index)^ := Resolved[Index];
end;

function FormatMissing(const AMissing: TShimPathArray): string;
const
  SHOWN = 10;
var
  Index: Integer;
begin
  Result := '';
  for Index := 0 to High(AMissing) do
  begin
    if Index = SHOWN then
    begin
      Result := Result + Format(', ... (%d more)', [Length(AMissing) - SHOWN]);
      Break;
    end;
    if Index > 0 then
      Result := Result + ', ';
    Result := Result + AMissing[Index];
  end;
end;

{ The SDK line is reported, not guessed. Measured reasons it cannot be a hard
  gate on every machine: 10.1 ships HIP 7.16, 7.2 and 10.1 share
  libamdhip64.so.7, the 10.x layout has no .info/version, and a
  local build.sh without ROCM_VERSION stamps the HIP version. So: compare when
  the caller pins PASROCM_EXPECT_ROCM, warn when the stamp maps to a different
  HIP major, and let PASROCM_STRICT_ROCM turn the warning into an error. }
function CheckRocmLine(const AShimVersion, AExpectPrefix: string;
  AHipRuntime: Int32): string;
var
  SdkMajor, ExpectedMajor, ActualMajor: Integer;
begin
  Result := '';

  if AExpectPrefix <> '' then
  begin
    if Pos(UpperCase(AExpectPrefix), UpperCase(AShimVersion)) <> 1 then
      Result := Format('shim reports SDK "%s" but %s requires "%s"',
        [AShimVersion, SHIM_ENV_EXPECT_ROCM, AExpectPrefix]);
    Exit;
  end;

  if (AShimVersion = '') or SameText(AShimVersion, 'unknown') then
  begin
    Result := 'shim carries no SDK stamp (built without ROCM_VERSION)';
    Exit;
  end;

  SdkMajor := VersionMajor(AShimVersion);
  if (SdkMajor < 6) or (SdkMajor > 10) then
    Exit;  { not an SDK line this table knows; nothing to compare }

  if SdkMajor >= 10 then
    ExpectedMajor := 7
  else
    ExpectedMajor := SdkMajor;

  ActualMajor := AHipRuntime div 10000000;
  if (ActualMajor > 0) and (ActualMajor <> ExpectedMajor) then
    Result := Format('shim was built for SDK %s (HIP %d) but the runtime is HIP %d',
      [AShimVersion, ExpectedMajor, ActualMajor]);
end;

{ The hash this library is expected to have, or '' when nobody asked for one.
  PASROCM_SHIM_SHA256 wins; otherwise a `<file>.sha256` sidecar next to the
  library, which is what a release package ships. A local build has no sidecar,
  so building from source is unaffected. }
function ExpectedShimSHA256(const APath: string): string;
var
  FromEnv: string;
  Sidecar: TextFile;
  Line: string;
  Sep: Integer;
begin
  FromEnv := Trim(EnvStr(SHIM_ENV_SHA256));
  if FromEnv <> '' then
    Exit(LowerCase(FromEnv));

  Result := '';
  if (APath = '') or (not FileExists(APath + '.sha256')) then
    Exit;

  AssignFile(Sidecar, APath + '.sha256');
  Reset(Sidecar);
  try
    if Eof(Sidecar) then
      Exit;
    ReadLn(Sidecar, Line);
  finally
    CloseFile(Sidecar);
  end;

  { Accept both "<hash>" and sha256sum's "<hash>  <name>". }
  Line := Trim(Line);
  Sep := Pos(' ', Line);
  if Sep = 0 then
    Sep := Pos(#9, Line);
  if Sep > 0 then
    Line := Copy(Line, 1, Sep - 1);
  Result := LowerCase(Trim(Line));
end;

function PasROCmShimLoad: Boolean;
var
  Handle: TLibHandle;
  Candidates: TShimPathArray;
  Missing: TShimPathArray;
  Index: Integer;
  Chosen: string;
  Explicit: string;
  Gfx: string;
  ExpectGfx: string;
  DeviceError: string;
  DeviceIndex: Int32;
  DeviceWarp: Int32;
  DeviceKnown: Boolean;
  HipRuntime: Int32;
  ShimTarget: string;
  RocmVersion: string;
  BuildInfo: string;
  AbiVersion: Cardinal;
  CompiledWidth: Cardinal;
  Mask: Cardinal;
  Reported: Cardinal;
  CompatOk: Boolean;
  Verdict: TShimIdentityVerdict;
  IdentityMissing: string;
  WarningText: string;
  Searched: string;
  ExpectedHash: string;
  ActualHash: string;

  procedure Fail(AException: EShimError);
  begin
    if Handle <> NilHandle then
    begin
      UnloadLibrary(Handle);
      Handle := NilHandle;
    end;
    GState := ssFailed;
    GError := AException.Message;
    raise AException;
  end;

  procedure RequireIdentity(AIndex: Integer; const AName: string);
  var
    MissingName: string;
  begin
    if not BindOne(Handle, AIndex, MissingName) then
      IdentityMissing := IdentityMissing + ' ' + AName;
  end;

begin
  if GState = ssLoaded then
    Exit(True);

  GLock.Enter;
  try
    if GState = ssLoaded then
      Exit(True);
    if GState = ssFailed then
      raise EShimError.Create(GError);

    GError := '';
    GInfo := Default(TShimLoadInfo);
    GInfo.DeviceIndex := -1;
    GInfo.HipRuntimeVersion := -1;
    Handle := NilHandle;

    { --- 1. Which GPU? ------------------------------------------------ }
    ExpectGfx := PasROCmNormalizeGfx(EnvStr(SHIM_ENV_EXPECT_GFX));
    DeviceKnown := ExpectGfx = '';
    if DeviceKnown then
    begin
      if not DetectDevice(Gfx, DeviceWarp, DeviceIndex, DeviceError) then
        Fail(EShimLoadFailed.Create('PasROCm: ' + DeviceError));
    end
    else
    begin
      Gfx := ExpectGfx;
      DeviceWarp := 0;
      DeviceIndex := -1;
    end;

    if hipRuntimeGetVersion(HipRuntime) <> hipSuccess then
      HipRuntime := -1;

    { --- 2. Find and open --------------------------------------------- }
    Explicit := EnvStr(SHIM_ENV_FILE);
    if (Explicit <> '') and (not FileExists(ExpandFileName(Explicit))) then
      Fail(EShimLibraryNotFound.CreateFmt('PasROCm: %s=%s does not exist',
        [SHIM_ENV_FILE, Explicit]));

    Candidates := PasROCmShimCandidatePaths(Gfx);
    Chosen := '';
    Searched := '';
    for Index := 0 to High(Candidates) do
    begin
      { An explicit file is authoritative: no silent fallback. }
      if (Explicit <> '') and (Index > 0) then
        Break;
      { Named paths must exist; a bare file name is left to dlopen, which
        searches LD_LIBRARY_PATH and the system directories. }
      if (ExtractFilePath(Candidates[Index]) <> '')
        and (not FileExists(Candidates[Index])) then
      begin
        Searched := Searched + LineEnding + '  ' + Candidates[Index]
          + '  (no such file)';
        Continue;
      end;
      Handle := LoadLibrary(Candidates[Index]);
      if Handle <> NilHandle then
      begin
        Chosen := Candidates[Index];
        Break;
      end;
      Searched := Searched + LineEnding + '  ' + Candidates[Index]
        + '  (' + GetLoadErrorStr + ')';
    end;

    if Handle = NilHandle then
      Fail(EShimLibraryNotFound.CreateFmt(
        'PasROCm: no shim for %s was found.%s'
        + 'Build one with: (cd src/rocprim && GFX_ARCH=%s REQUIRE_ARCH=1 ./build.sh)%s'
        + 'Or point %s at the directory holding %s.%sSearched:%s',
        [Gfx, LineEnding, Gfx, LineEnding, SHIM_ENV_DIR,
         PasROCmShimLibraryFileName(Gfx), LineEnding, Searched]));

    { --- 2b. Integrity pin (optional) ----------------------------------
      A release package ships `<file>.so.sha256`; PASROCM_SHIM_SHA256 overrides
      it. Neither exists for a local build, so nothing is hashed then. This
      closes the hole the identity gate cannot: a library built for the right
      target by someone else, earlier in the search path. }
    ExpectedHash := ExpectedShimSHA256(Chosen);
    if ExpectedHash <> '' then
    begin
      if not FileExists(Chosen) then
        Fail(EShimIdentityError.CreateFmt(
          'PasROCm: a SHA256 was requested for %s, but the loader only has the '
          + 'bare name "%s" (dlopen resolved it). Name the file with %s or its '
          + 'directory with %s so it can be hashed.',
          [Gfx, Chosen, SHIM_ENV_FILE, SHIM_ENV_DIR]));
      ActualHash := PasROCmShimFileSHA256(Chosen);
      if ActualHash <> ExpectedHash then
        Fail(EShimIdentityError.CreateFmt(
          'PasROCm: %s does not match its expected SHA256.%s'
          + '  expected: %s%s  actual:   %s%s'
          + 'Refusing to load it. If you built this library yourself, unset %s '
          + 'and remove the .sha256 sidecar.',
          [Chosen, LineEnding, ExpectedHash, LineEnding, ActualHash,
           LineEnding, SHIM_ENV_SHA256]));
    end;

    { --- 3. Identity, through the pure exports only -------------------- }
    IdentityMissing := '';
    RequireIdentity(ROCPRIM_SHIM_IDX_TARGET, 'pasrocm_shim_target');
    RequireIdentity(ROCPRIM_SHIM_IDX_WAVEFRONT_SIZE, 'pasrocm_shim_wavefront_size');
    RequireIdentity(ROCPRIM_SHIM_IDX_ROCM_VERSION, 'pasrocm_shim_rocm_version');
    RequireIdentity(ROCPRIM_SHIM_IDX_WIDTHS_MASK, 'pasrocm_shim_wavefront_widths_mask');
    RequireIdentity(ROCPRIM_SHIM_IDX_WAVEFRONT_COMPATIBLE, 'pasrocm_wavefront_compatible');
    RequireIdentity(ROCPRIM_SHIM_IDX_ABI_VERSION, 'pasrocm_shim_abi_version');
    RequireIdentity(ROCPRIM_SHIM_IDX_BUILD_INFO, 'pasrocm_shim_build_info');
    if IdentityMissing <> '' then
      Fail(EShimBindError.CreateFmt(
        'PasROCm: %s is not a PasROCm shim: it does not export%s',
        [Chosen, IdentityMissing]));

    ShimTarget := string(rocprim_shim_api.__pasrocm_shim_target());
    CompiledWidth := rocprim_shim_api.__pasrocm_shim_wavefront_size();
    Mask := rocprim_shim_api.__pasrocm_shim_wavefront_widths_mask();
    RocmVersion := string(rocprim_shim_api.__pasrocm_shim_rocm_version());
    AbiVersion := rocprim_shim_api.__pasrocm_shim_abi_version();
    BuildInfo := string(rocprim_shim_api.__pasrocm_shim_build_info());

    { Hard gate, before anything else is interpreted: a different ABI means
      the prototypes this executable was generated from do not describe this
      library. 0 means the library reports none, which is also refused. }
    if not PasROCmAbiVersionAccepted(AbiVersion, ROCPRIM_SHIM_ABI_VERSION) then
      Fail(EShimIdentityError.CreateFmt(
        'PasROCm: %s reports ABI %d but this build expects %d%s'
        + 'Install the matching PasROCm units and shim package.%s',
        [Chosen, AbiVersion, ROCPRIM_SHIM_ABI_VERSION, LineEnding, LineEnding]));

    { No device (PASROCM_EXPECT_GFX): the mask pins the hardware width when it
      is single-valued, exactly the way the C identity self-test does. }
    if DeviceKnown then
      Reported := Cardinal(DeviceWarp)
    else if Mask = 1 then
      Reported := 32
    else if Mask = 2 then
      Reported := 64
    else
      Reported := CompiledWidth;

    CompatOk := rocprim_shim_api.__pasrocm_wavefront_compatible(
      CompiledWidth, Reported, Mask) <> 0;

    { --- 4. Refuse the wrong library ---------------------------------- }
    Verdict := PasROCmClassifyIdentity(ShimTarget, Gfx, CompiledWidth,
      Reported, Mask, CompatOk);
    case Verdict of
      ivTargetMismatch:
        Fail(EShimIdentityError.CreateFmt(
          'PasROCm: %s was built for %s but this device is %s.%s'
          + 'Load the shim for %s (one library per GPU target).',
          [Chosen, ShimTarget, Gfx, LineEnding, Gfx]));
      ivBadCompiledWidth:
        Fail(EShimIdentityError.CreateFmt(
          'PasROCm: %s reports wavefront size %d; only 32 or 64 are valid',
          [Chosen, CompiledWidth]));
      ivBadMask:
        Fail(EShimIdentityError.CreateFmt(
          'PasROCm: %s reports widths mask %d for a wave%d build, which the '
          + 'hardware it targets cannot run', [Chosen, Mask, CompiledWidth]));
      ivWidthMismatch:
        Fail(EShimIdentityError.CreateFmt(
          'PasROCm: %s was compiled for wave%d but the device reports wave%d '
          + '(widths mask %d). This is the wrong library for this GPU.',
          [Chosen, CompiledWidth, Reported, Mask]));
      ivOk:
        ;  { the only path that continues }
    end;

    WarningText := CheckRocmLine(RocmVersion, EnvStr(SHIM_ENV_EXPECT_ROCM),
      HipRuntime);
    if (WarningText <> '') and (EnvStr(SHIM_ENV_STRICT_ROCM) <> '') then
      Fail(EShimIdentityError.CreateFmt('PasROCm: %s: %s',
        [Chosen, WarningText]));

    { --- 5. Resolve the whole table, atomically ----------------------- }
    BindAll(Handle, Missing);
    if Length(Missing) > 0 then
      Fail(EShimBindError.CreateFmt(
        'PasROCm: %s is missing %d of %d symbols: %s',
        [Chosen, Length(Missing), ROCPRIM_SHIM_SYMBOL_COUNT,
         FormatMissing(Missing)]));

    GHandle := Handle;
    Handle := NilHandle;
    GInfo.LibraryPath := Chosen;
    GInfo.Target := ShimTarget;
    GInfo.RocmVersion := RocmVersion;
    GInfo.WavefrontSize := CompiledWidth;
    GInfo.WidthsMask := Mask;
    GInfo.AbiVersion := AbiVersion;
    GInfo.BuildInfo := BuildInfo;
    GInfo.DeviceGfx := Gfx;
    GInfo.DeviceIndex := DeviceIndex;
    GInfo.DeviceWarpSize := DeviceWarp;
    GInfo.HipRuntimeVersion := HipRuntime;
    GInfo.DeviceOverridden := not DeviceKnown;
    GInfo.Warnings := WarningText;
    GState := ssLoaded;
    Result := True;
  finally
    GLock.Leave;
  end;
end;

function PasROCmShimEnsureLoaded: Boolean;
begin
  { Fast path, taken on every shim call. Safe on x86_64: an aligned 32-bit load
    is atomic, and the store that publishes ssLoaded is ordered after the slot
    writes, so a reader that sees ssLoaded also sees a complete table (TSO).
    A weakly ordered target (aarch64) would need a barrier here, or simply the
    unconditional PasROCmShimLoad -- an uncontended critical section is
    negligible next to a kernel launch. }
  if GState = ssLoaded then
    Exit(True);
  Result := PasROCmShimLoad;
end;

procedure PasROCmShimUnload;
var
  Index: Integer;
begin
  GLock.Enter;
  try
    if GHandle <> NilHandle then
    begin
      for Index := 0 to ROCPRIM_SHIM_SYMBOL_COUNT - 1 do
        if rocprim_shim_symbol_slot(Index) <> nil then
          rocprim_shim_symbol_slot(Index)^ := nil;
      UnloadLibrary(GHandle);
      GHandle := NilHandle;
    end;
    GState := ssIdle;
    GError := '';
    GInfo := Default(TShimLoadInfo);
  finally
    GLock.Leave;
  end;
end;

function PasROCmShimLoaded: Boolean;
begin
  Result := GState = ssLoaded;
end;

function PasROCmShimInfo: TShimLoadInfo;
begin
  { Copies strings, so it must not overlap a write. The lock is uncontended
    once the library is loaded. }
  GLock.Enter;
  try
    Result := GInfo;
  finally
    GLock.Leave;
  end;
end;

function PasROCmShimLastError: string;
begin
  GLock.Enter;
  try
    Result := GError;
  finally
    GLock.Leave;
  end;
end;

initialization
  GLock := TCriticalSection.Create;

finalization
  { The library is deliberately not unloaded here: a stream may still be in
    flight and the process is exiting anyway. PasROCmShimUnload is for tests. }
  GLock.Free;

end.
