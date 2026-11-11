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

program TestRunner;

{$mode objfpc}{$H+}

uses
  Classes, math, consoletestrunner, fpcunit

  { Tests }

  , Test_rocfft
  , Test_rocblas_base     // necessary for all rocblas tests
  , Test_rocblas_lvl1
  , Test_rocblas_lvl1_64
  , Test_rocblas_lvl1_batched
  , Test_rocblas_lvl1_batched_64
  , Test_rocblas_lvl1_strided_batched
  , Test_rocblas_lvl1_strided_batched_64
  , Test_rocblas_lvl2
  , Test_rocblas_lvl2_64
  , Test_rocblas_lvl2_batched
  , Test_rocblas_lvl2_batched_64
  , Test_rocblas_lvl2_strided_batched
  , Test_rocblas_lvl2_strided_batched_64
  , Test_rocblas_lvl3
  , Test_rocblas_lvl3_64
  , Test_rocblas_lvl3_batched
  , Test_rocblas_lvl3_batched_64
  , Test_rocblas_lvl3_strided_batched
  , Test_rocblas_lvl3_strided_batched_64
  , Test_miopen
  , Test_rocrand
  , Test_rocsolver
  , Test_rocsparse 
  , Test_hip_runtime
  ;

type

  { TMyTestRunner }

  TMyTestRunner = class(TTestRunner)
  protected
  // override the protected methods of TTestRunner to customize its behavior
  end;

var
  Application: TMyTestRunner;

begin
    SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow, exUnderflow, exPrecision]);
  DefaultRunAllTests:=True;
  DefaultFormat:=fplain;
  Application := TMyTestRunner.Create(nil);
  Application.Initialize;
  Application.Title := 'FPCUnit Console test runner';
  Application.Run;
  Application.Free;
end.
