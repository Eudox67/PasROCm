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

unit Test_rocblas;

{$mode objfpc}{$H+}
{$linklib rocblas}
{$DEFINE ROCM_6_3_2}

interface

uses
  fpcunit, testregistry, rocblas, hip_runtime, SysUtils. rocm_test_utils;


type
  { TTestRocBLAS_Base }
  // Base class for rocBLAS tests. Not registered as a test itself.
  // Handles common setup and teardown for the rocBLAS handle.
  TTestRocBLAS_Base = class(TTestCase)
  protected
    handle: rocblas_handle;
    procedure Setup; override;
    procedure TearDown; override;
  end;


implementation

procedure TTestRocBLAS_Base.Setup;
begin
  AssertEquals('rocblas_create_handle should return success', Integer(rocblas_status_success), Integer(rocblas_create_handle(handle)));
end;

procedure TTestRocBLAS_Base.TearDown;
begin
  AssertEquals('rocblas_destroy_handle should return success', Integer(rocblas_status_success), Integer(rocblas_destroy_handle(handle)));
end;

initialization
  RegisterTest(TTestRocBLAS_Base);

end.
