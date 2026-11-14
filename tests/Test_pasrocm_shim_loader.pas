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

{ Tests for rocprim_shim_loader.

  Two layers:

  * the pure policy (name normalization, candidate order, identity verdicts) --
    these run everywhere, with no GPU and no shim;
  * the real load -- these need a built libpasrocm_shim_<gfx>.so and a device,
    and Ignore() with the reason when either is missing.

  The negative load tests are the point of the whole unit: they prove that a
  library built for another GPU, or a library that is not a PasROCm shim, is
  refused with a named exception instead of being used. They need no second
  GPU and no second binary: PASROCM_SHIM points at the good library while
  PASROCM_EXPECT_GFX claims a different target.
}

unit Test_pasrocm_shim_loader;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, fpcunit, testregistry,
  rocprim_shim_loader, rocprim_shim_api, rocprim_bindings;

type
  TTestShimLoader = class(TTestCase)
  private
    function LibraryFileNameFor(const AGfx: string): string;
    procedure UnsetEnv(const AName: string);
    procedure SetEnv(const AName, AValue: string);
    { Load the shim for this machine, or Ignore() the test with a reason. }
    function EnsureShimOrIgnore: Boolean;
  published
    { Pure policy -- no GPU, no library. }
    procedure TestNormalizeGfxStripsFeatures;
    procedure TestNormalizeGfxOfEmptyAndNoise;
    procedure TestLibraryFileNameCarriesTarget;
    procedure TestCandidatePathsPreferExplicitDirectory;
    procedure TestCandidatePathsPreferExplicitFile;
    procedure TestCandidatePathsEndWithBareName;
    procedure TestIdentityAcceptsMatchingTarget;
    procedure TestIdentityAcceptsDualWidthHardware;
    procedure TestIdentityRejectsTargetMismatch;
    procedure TestIdentityRejectsImpossibleCompiledWidth;
    procedure TestIdentityRejectsBadMask;
    procedure TestIdentityRejectsCompatFailure;

    { Real load -- needs a shim and a GPU. }
    procedure TestLoadsShimForThisDevice;
    procedure TestEverySlotIsBound;
    procedure TestEnsureLoadedIsIdempotent;
    procedure TestRefusesForeignTarget;
    procedure TestRefusesNonShimLibrary;
    procedure TestUnloadAllowsReload;
  end;

implementation

{$IFDEF UNIX}
{ This FPC tree exposes no setenv() wrapper, so bind libc directly. Test-only:
  the loader reads the environment, and the negative tests need to shape it. }
function libc_setenv(name, value: PAnsiChar; overwrite: LongInt): LongInt;
  cdecl; external 'c' name 'setenv';
function libc_unsetenv(name: PAnsiChar): LongInt;
  cdecl; external 'c' name 'unsetenv';
{$ENDIF}

const
  { Any glibc system has one of these; used only to prove that a real library
    which is not a PasROCm shim is rejected for the right reason. }
  FOREIGN_LIBRARY: array[0..2] of string = (
    '/lib/x86_64-linux-gnu/libm.so.6',
    '/lib64/libm.so.6',
    '/usr/lib64/libm.so.6'
  );

procedure TTestShimLoader.UnsetEnv(const AName: string);
begin
  {$IFDEF UNIX}
  libc_unsetenv(PAnsiChar(AnsiString(AName)));
  {$ENDIF}
end;

procedure TTestShimLoader.SetEnv(const AName, AValue: string);
begin
  {$IFDEF UNIX}
  libc_setenv(PAnsiChar(AnsiString(AName)), PAnsiChar(AnsiString(AValue)), 1);
  {$ENDIF}
end;

function TTestShimLoader.LibraryFileNameFor(const AGfx: string): string;
begin
  Result := PasROCmShimLibraryFileName(AGfx);
end;

function TTestShimLoader.EnsureShimOrIgnore: Boolean;
begin
  Result := False;
  { A failure latched by an earlier test would mask the real reason here. }
  if not PasROCmShimLoaded then
    PasROCmShimUnload;
  try
    PasROCmShimEnsureLoaded;
    Result := True;
  except
    on E: EShimLibraryNotFound do
      Ignore('no shim for this machine: ' + E.Message);
    on E: EShimLoadFailed do
      Ignore('no usable GPU for the loader: ' + E.Message);
  end;
end;

{ --------------------------------------------------------------------------
  Pure policy
  -------------------------------------------------------------------------- }

procedure TTestShimLoader.TestNormalizeGfxStripsFeatures;
begin
  AssertEquals('feature suffixes are dropped', 'gfx90a',
    PasROCmNormalizeGfx('gfx90a:sramecc+:xnack-'));
  AssertEquals('lowercase', 'gfx1100', PasROCmNormalizeGfx('GFX1100'));
  AssertEquals('trimmed', 'gfx942', PasROCmNormalizeGfx('  gfx942  '));
  AssertEquals('single feature', 'gfx1030', PasROCmNormalizeGfx('gfx1030:xnack-'));
end;

procedure TTestShimLoader.TestNormalizeGfxOfEmptyAndNoise;
begin
  AssertEquals('empty stays empty', '', PasROCmNormalizeGfx(''));
  AssertEquals('blank stays empty', '', PasROCmNormalizeGfx('   '));
  AssertEquals('only a feature suffix', '', PasROCmNormalizeGfx(':xnack+'));
end;

procedure TTestShimLoader.TestLibraryFileNameCarriesTarget;
begin
  {$IFDEF MSWINDOWS}
  AssertEquals('windows name', 'pasrocm_shim_gfx1100.dll',
    PasROCmShimLibraryFileName('gfx1100'));
  {$ELSE}
  AssertEquals('unix name', 'libpasrocm_shim_gfx1100.so',
    PasROCmShimLibraryFileName('gfx1100'));
  AssertEquals('CDNA name', 'libpasrocm_shim_gfx942.so',
    PasROCmShimLibraryFileName('gfx942'));
  {$ENDIF}
end;

procedure TTestShimLoader.TestCandidatePathsPreferExplicitDirectory;
var
  Paths: TShimPathArray;
begin
  SetEnv(SHIM_ENV_DIR, '/tmp/pasrocm-shim-dir');
  try
    Paths := PasROCmShimCandidatePaths('gfx1100');
    AssertTrue('at least one candidate', Length(Paths) > 0);
    AssertEquals('PASROCM_SHIM_DIR wins',
      IncludeTrailingPathDelimiter('/tmp/pasrocm-shim-dir')
        + LibraryFileNameFor('gfx1100'), Paths[0]);
  finally
    UnsetEnv(SHIM_ENV_DIR);
  end;
end;

procedure TTestShimLoader.TestCandidatePathsPreferExplicitFile;
var
  Paths: TShimPathArray;
begin
  SetEnv(SHIM_ENV_FILE, '/tmp/libpasrocm_shim_gfx1100.so');
  try
    Paths := PasROCmShimCandidatePaths('gfx1100');
    AssertEquals('PASROCM_SHIM outranks the directory',
      '/tmp/libpasrocm_shim_gfx1100.so', Paths[0]);
  finally
    UnsetEnv(SHIM_ENV_FILE);
  end;
end;

procedure TTestShimLoader.TestCandidatePathsEndWithBareName;
var
  Paths: TShimPathArray;
begin
  Paths := PasROCmShimCandidatePaths('gfx1100');
  AssertTrue('candidates exist', Length(Paths) > 0);
  { The bare name is what makes LD_LIBRARY_PATH keep working. }
  AssertEquals('last candidate is the bare file name',
    LibraryFileNameFor('gfx1100'), Paths[High(Paths)]);
end;

procedure TTestShimLoader.TestIdentityAcceptsMatchingTarget;
begin
  AssertEquals('wave32 on wave32-only', Ord(ivOk),
    Ord(PasROCmClassifyIdentity('gfx1100', 'gfx1100', 32, 32, 1, True)));
  AssertEquals('wave64 on CDNA', Ord(ivOk),
    Ord(PasROCmClassifyIdentity('gfx942', 'gfx942', 64, 64, 2, True)));
end;

procedure TTestShimLoader.TestIdentityAcceptsDualWidthHardware;
begin
  { RDNA4 class: mask 3 accepts either width, and the reported width does not
    have to equal the compiled width (DESIGN.md section 9). }
  AssertEquals('wave64 build, device reports 32', Ord(ivOk),
    Ord(PasROCmClassifyIdentity('gfx1250', 'gfx1250', 64, 32, 3, True)));
  AssertEquals('wave32 build, device reports 64', Ord(ivOk),
    Ord(PasROCmClassifyIdentity('gfx1250', 'gfx1250', 32, 64, 3, True)));
end;

procedure TTestShimLoader.TestIdentityRejectsTargetMismatch;
begin
  AssertEquals('wrong GPU', Ord(ivTargetMismatch),
    Ord(PasROCmClassifyIdentity('gfx942', 'gfx1100', 64, 64, 2, True)));
  AssertEquals('case does not matter', Ord(ivOk),
    Ord(PasROCmClassifyIdentity('GFX1100', 'gfx1100', 32, 32, 1, True)));
end;

procedure TTestShimLoader.TestIdentityRejectsImpossibleCompiledWidth;
begin
  AssertEquals('compiled 48', Ord(ivBadCompiledWidth),
    Ord(PasROCmClassifyIdentity('gfx1100', 'gfx1100', 48, 32, 1, True)));
  AssertEquals('compiled 0', Ord(ivBadCompiledWidth),
    Ord(PasROCmClassifyIdentity('gfx1100', 'gfx1100', 0, 32, 1, True)));
end;

procedure TTestShimLoader.TestIdentityRejectsBadMask;
begin
  AssertEquals('no allowed widths', Ord(ivBadMask),
    Ord(PasROCmClassifyIdentity('gfx1100', 'gfx1100', 32, 32, 0, True)));
  AssertEquals('mask out of range', Ord(ivBadMask),
    Ord(PasROCmClassifyIdentity('gfx1100', 'gfx1100', 32, 32, 9, True)));
  AssertEquals('wave64 build, wave32-only hardware', Ord(ivBadMask),
    Ord(PasROCmClassifyIdentity('gfx1100', 'gfx1100', 64, 32, 1, True)));
  AssertEquals('wave32 build, wave64-only hardware', Ord(ivBadMask),
    Ord(PasROCmClassifyIdentity('gfx942', 'gfx942', 32, 64, 2, True)));
end;

procedure TTestShimLoader.TestIdentityRejectsCompatFailure;
begin
  { The shim's own predicate said no: never second-guess it in Pascal. }
  AssertEquals('compat false is fatal', Ord(ivWidthMismatch),
    Ord(PasROCmClassifyIdentity('gfx1100', 'gfx1100', 32, 32, 1, False)));
  AssertEquals('nonsense reported width', Ord(ivWidthMismatch),
    Ord(PasROCmClassifyIdentity('gfx1100', 'gfx1100', 32, 48, 3, True)));
end;

{ --------------------------------------------------------------------------
  Real load
  -------------------------------------------------------------------------- }

procedure TTestShimLoader.TestLoadsShimForThisDevice;
var
  Info: TShimLoadInfo;
begin
  if not EnsureShimOrIgnore then
    Exit;
  Info := PasROCmShimInfo;
  AssertTrue('library path is remembered', Info.LibraryPath <> '');
  AssertTrue('target is known', Info.Target <> '');
  AssertEquals('the public wrapper and the info agree', Info.Target,
    string(pasrocm_shim_target()));
  AssertTrue('wavefront is 32 or 64',
    (Info.WavefrontSize = 32) or (Info.WavefrontSize = 64));
  AssertTrue('widths mask is 1..3',
    (Info.WidthsMask >= 1) and (Info.WidthsMask <= 3));
  if not Info.DeviceOverridden then
  begin
    AssertTrue('device gfx was detected', Info.DeviceGfx <> '');
    AssertEquals('the library matches the detected device', Info.Target,
      Info.DeviceGfx);
  end;
end;

procedure TTestShimLoader.TestEverySlotIsBound;
var
  Index: Integer;
begin
  if not EnsureShimOrIgnore then
    Exit;
  AssertEquals('table size', 149, ROCPRIM_SHIM_SYMBOL_COUNT);
  for Index := 0 to ROCPRIM_SHIM_SYMBOL_COUNT - 1 do
  begin
    AssertTrue('name at ' + IntToStr(Index) + ' is set',
      rocprim_shim_symbol_name(Index) <> nil);
    AssertTrue('slot for ' + string(rocprim_shim_symbol_name(Index)) + ' is bound',
      rocprim_shim_symbol_slot(Index)^ <> nil);
  end;
end;

procedure TTestShimLoader.TestEnsureLoadedIsIdempotent;
var
  First, Second: TShimLoadInfo;
begin
  if not EnsureShimOrIgnore then
    Exit;
  First := PasROCmShimInfo;
  AssertTrue('second ensure returns true', PasROCmShimEnsureLoaded);
  AssertTrue('still loaded', PasROCmShimLoaded);
  Second := PasROCmShimInfo;
  AssertEquals('same library', First.LibraryPath, Second.LibraryPath);
  AssertEquals('same target', First.Target, Second.Target);
end;

procedure TTestShimLoader.TestRefusesForeignTarget;
var
  LibPath: string;
  Raised: Boolean;
begin
  if not EnsureShimOrIgnore then
    Exit;
  LibPath := PasROCmShimInfo.LibraryPath;

  PasROCmShimUnload;
  SetEnv(SHIM_ENV_FILE, LibPath);
  SetEnv(SHIM_ENV_EXPECT_GFX, 'gfx9999');
  try
    Raised := False;
    try
      PasROCmShimEnsureLoaded;
    except
      on E: EShimIdentityError do
        Raised := True;
    end;
    AssertTrue('a library for another GPU raises EShimIdentityError', Raised);
  finally
    UnsetEnv(SHIM_ENV_FILE);
    UnsetEnv(SHIM_ENV_EXPECT_GFX);
    PasROCmShimUnload;  { clear the latched failure }
  end;
end;

procedure TTestShimLoader.TestRefusesNonShimLibrary;
var
  Candidate: string;
  Index: Integer;
  Raised: Boolean;
begin
  Candidate := '';
  for Index := 0 to High(FOREIGN_LIBRARY) do
    if FileExists(FOREIGN_LIBRARY[Index]) then
    begin
      Candidate := FOREIGN_LIBRARY[Index];
      Break;
    end;
  if Candidate = '' then
  begin
    Ignore('no system libm.so.6 available as a foreign library');
    Exit;
  end;

  PasROCmShimUnload;
  SetEnv(SHIM_ENV_FILE, Candidate);
  try
    Raised := False;
    try
      PasROCmShimEnsureLoaded;
    except
      on E: EShimBindError do
        Raised := True;
    end;
    AssertTrue('a library without the identity exports raises EShimBindError',
      Raised);
  finally
    UnsetEnv(SHIM_ENV_FILE);
    PasROCmShimUnload;
  end;
end;

procedure TTestShimLoader.TestUnloadAllowsReload;
var
  Before, Reloaded: TShimLoadInfo;
begin
  if not EnsureShimOrIgnore then
    Exit;
  Before := PasROCmShimInfo;

  PasROCmShimUnload;
  AssertFalse('unload clears the loaded state', PasROCmShimLoaded);

  AssertTrue('reload succeeds', PasROCmShimEnsureLoaded);
  Reloaded := PasROCmShimInfo;
  AssertEquals('same library after reload', Before.LibraryPath,
    Reloaded.LibraryPath);
  AssertEquals('same target after reload', Before.Target, Reloaded.Target);
end;

initialization
  RegisterTest(TTestShimLoader);

end.
