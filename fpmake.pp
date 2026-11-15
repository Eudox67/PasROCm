program fpmake;

{$mode objfpc}{$H+}

uses
{$ifdef unix}
  cthreads,
{$endif}
  SysUtils, fpmkunit;

{ The project version lives in one file at the repository root, next to this
  program. Reading it here keeps fpmake, build.sh and ci/manifest.sh reporting
  the same number instead of each carrying its own copy. }
function ProjectVersion: string;
var
  Candidates: array[0..1] of string;
  I: Integer;
  VersionFile: TextFile;
  Line: string;
begin
  Candidates[0] := 'VERSION';
  Candidates[1] := IncludeTrailingPathDelimiter(ExtractFilePath(ParamStr(0))) + 'VERSION';
  for I := 0 to High(Candidates) do
    if FileExists(Candidates[I]) then
    begin
      AssignFile(VersionFile, Candidates[I]);
      Reset(VersionFile);
      try
        ReadLn(VersionFile, Line);
      finally
        CloseFile(VersionFile);
      end;
      Result := Trim(Line);
      if Result <> '' then
        Exit;
    end;
  WriteLn('fpmake: cannot read VERSION -- run this from the repository root');
  Halt(1);
end;

var
  P: TPackage;
begin
  with Installer do
  begin
    P := AddPackage('pasrocm');
    P.Version := ProjectVersion;
    P.Author := 'PasROCm Contributors';
    P.License := 'MIT';
    P.Description := 'Pascal bindings for AMD ROCm libraries';

    P.OSes := [linux];
    P.CPUs := [x86_64];

    // UnitPath feeds -Fu, SourcePath locates the files named in AddUnit, and
    // IncludePath feeds -Fi for src/miopen_defs.inc.
    P.SourcePath.Add('src');
    P.SourcePath.Add('src/rocprim');
    P.UnitPath.Add('src');
    P.UnitPath.Add('src/rocprim');
    P.IncludePath.Add('src');

    // The units carry no {$mode} directive of their own: -Mobjfpc is required
    // (they use e.g. default parameter values and advanced records).
    P.Options.Add('-Mobjfpc');

    // fpmkunit injects an implicit dependency on the "rtl" package. Resolving
    // an installed package needs a per-package unit-config file in the compiler
    // installation (<unitdir>/rtl/rtl.fpc); many FPC installs only ship the
    // fppkg-style "Package.fpc", and the lookup then aborts with
    //   Could not find unit directory for dependency package "rtl"
    // The compiler always knows where its own RTL is, so the dependency is
    // dropped here and this package needs no external package resolution.
    P.Dependencies.Clear;

    // --- HIP runtime and utilities ---
    P.Targets.AddUnit('hip_common.pas');
    P.Targets.AddUnit('hip_error.pas');
    P.Targets.AddUnit('hip_types.pas');
    P.Targets.AddUnit('hip_vector_types.pas');
    P.Targets.AddUnit('hip_runtime.pas');
    P.Targets.AddUnit('hiprtc.pas');

    // --- ROCm compute libraries ---
    P.Targets.AddUnit('rocblas.pas');
    P.Targets.AddUnit('rocsparse.pas');
    P.Targets.AddUnit('rocsolver.pas');
    P.Targets.AddUnit('rocrand.pas');
    P.Targets.AddUnit('rocfft.pas');
    P.Targets.AddUnit('miopen.pas');
    P.Targets.AddUnit('rccl.pas');

    // --- rocPRIM (facade + component units) ---
    P.Targets.AddUnit('rocprim.pas');
    P.Targets.AddUnit('hip_bindings.pas');
    P.Targets.AddUnit('hip_specialized.pas');
    P.Targets.AddUnit('rocprim_shim_api.pas');
    P.Targets.AddUnit('rocprim_shim_loader.pas');
    P.Targets.AddUnit('rocprim_bindings.pas');
    P.Targets.AddUnit('rocprim_block.pas');
    P.Targets.AddUnit('rocprim_core.pas');
    P.Targets.AddUnit('rocprim_device.pas');
    P.Targets.AddUnit('rocprim_intrinsics.pas');
    P.Targets.AddUnit('rocprim_iterator.pas');
    P.Targets.AddUnit('rocprim_thread.pas');
    P.Targets.AddUnit('rocprim_types.pas');
    P.Targets.AddUnit('rocprim_warp.pas');

    Run;
  end;
end.
