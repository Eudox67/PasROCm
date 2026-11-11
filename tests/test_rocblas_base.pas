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

unit Test_rocblas_base;

{$mode ObjFPC}{$H+}
{$linklib rocblas}

interface

uses
  SysUtils, fpcunit, testregistry, rocblas, hip_types, hip_runtime, ROCm_test_utils;

type
  { TTestRocBLAS_Base }
  // Handles common setup and teardown for the rocBLAS handle.
  // Adds a CheckEqualsRelative method
  TTestRocBLAS_Base = class(TTestCase)
  protected
    handle: rocblas_handle;

    {Use CheckEqualsRelative when:
    1. Comparing Results of Floating-Point Operations with Varying Magnitudes:
       If the expected result of your computation can be very small (e.g., 0.001) or very large (e.g., 1000000.0),
       a fixed absolute tolerance (like 0.001 or 1.0) becomes problematic. An absolute tolerance of 1.0 is
       extremely strict for a result of 0.001 (100% error allowed vs. ~0.1% error) but very loose for a result of
       1000000.0 (0.0001% error allowed vs. potentially acceptable 0.1% error). Relative tolerance scales with the
       magnitude.
    2. Testing Lower-Precision Arithmetic (like rocblas_half, rocblas_bfloat16):
       The amount of error often scales with the magnitude of the result. Using a relative tolerance (e.g., 1%, 0.1%)
       is a more natural way to express "allow for the typical error introduced by this precision".
    3. Comparing Against Highly Accurate Reference Calculations:
       If your test calculates an expected result using a highly accurate method (e.g., double precision on the CPU)
       and compares it to a result calculated using potentially less accurate methods (e.g., single precision on GPU,
       or iterative methods), the error might be small in relative terms but large in absolute terms depending on the
       result's magnitude.
    4. General Robustness for Floating-Point Comparisons: Even for standard Single or Double precision comparisons,
       using a small relative tolerance (e.g., 1e-6 for Single, 1e-12 for Double) can be more robust than trying to
       find a "one size fits all" absolute tolerance, especially if the range of test values is wide.}

    procedure CheckEqualsRelative(const expected, actual, relative_tolerance: Single; const msg: string = '');

  published
    procedure Setup; override;
    procedure TearDown; override;

  end;

implementation

procedure TTestRocBLAS_Base.Setup;
var
  blas_status: rocblas_status;
begin
  inherited SetUp;

  handle := nil; // Initialize

  // Correct call: Pass 'handle' directly because the parameter is 'var'
  blas_status := rocblas_create_handle(handle);

  AssertEquals('rocblas_create_handle should return success',
               Integer(rocblas_status_success),
               Integer(blas_status));

  // Optional check
   AssertTrue('rocblas_create_handle succeeded but handle is still nil', Assigned(handle));
end;

procedure TTestRocBLAS_Base.TearDown;
var
  blas_status: rocblas_status;
begin
  // It's good practice to check if the handle was actually created
  // before trying to destroy it, especially if SetUp could fail.
  if Assigned(handle) then
  begin
    blas_status := rocblas_destroy_handle(handle); // Destroy takes the handle value
    AssertEquals('rocblas_destroy_handle should return success',
                 Integer(rocblas_status_success),
                 Integer(blas_status));
    handle := nil; // Optional: Clear the handle after destruction
  end;
  inherited TearDown; // Call inherited last
end;

procedure TTestRocBLAS_Base.CheckEqualsRelative(const expected, actual, relative_tolerance: Single; const msg: string = '');
var
  relative_error: Single;
  errorMessage: String;
begin
  // Calculate relative error
  if expected <> 0.0 then
    relative_error := Abs((expected - actual) / expected)
  else
    // Handle case where expected result is zero
    // In this case, absolute error is the same as relative error scaled to 1.
    relative_error := Abs(expected - actual); // Or could use Abs(actual) if expected is exactly 0.

  // Construct error message
  errorMessage := Format('%s Relative Error: %.8f (Expected: %.8f, Actual: %.8f, Tolerance: %.8f)',
                         [msg, relative_error, expected, actual, relative_tolerance]);

  // Use AssertTrue to check the condition and fail the test if not met
  AssertTrue(errorMessage, relative_error <= relative_tolerance);
end;

initialization
  //RegisterTest(TTestRocBLAS_Base);

end.

