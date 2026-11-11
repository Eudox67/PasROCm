program fpmake;

{$mode objfpc}{$H+}

uses
{$ifdef unix}
  cthreads,
{$endif}
  fpmkunit;

var
  P: TPackage;
begin
  with Installer do
  begin
    P := AddPackage('pasrocm');
    P.Version := '1.0.0';
    P.Author := 'PasROCm Contributors';
    P.License := 'MIT / BSD-3-Clause';
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
