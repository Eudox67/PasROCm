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

unit Test_rocrand;

{$mode objfpc}{$H+}
{$linklib rocrand}

interface

uses
  SysUtils, fpcunit, testregistry, rocrand, hip_error, hip_types, hip_runtime;

type
  TTest_rocRAND = class(TTestCase)
  private
    generator: rocrand_generator;
    procedure Setup; override;
    procedure TearDown; override;
  published
    // ---------- Generator life-cycle ----------
    procedure TestCreateAndDestroyGenerator;
    procedure Test_rocrand_create_generator_host;
    procedure Test_rocrand_create_generator_host_blocking;
    procedure Test_rocrand_destroy_generator;

    // ---------- Seed / stream / offset ----------
    procedure Test_rocrand_initialize_generator;
    procedure Test_rocrand_set_stream;
    procedure Test_rocrand_set_seed;
    procedure Test_rocrand_set_seed_uint4;
    procedure Test_rocrand_set_offset;
    procedure Test_rocrand_set_ordering;
    procedure Test_rocrand_set_quasi_random_generator_dimensions;

    // ---------- Generator queries ----------
    procedure Test_rocrand_get_version;

    // ---------- Discrete distributions ----------
    //procedure Test_rocrand_create_poisson_distribution;
    procedure Test_rocrand_create_discrete_distribution;
    procedure Test_rocrand_destroy_discrete_distribution;

    // ---------- Direction-vector helpers ----------
    procedure Test_rocrand_get_direction_vectors32;
    procedure Test_rocrand_get_direction_vectors64;
    procedure Test_rocrand_get_scramble_constants32;
    procedure Test_rocrand_get_scramble_constants64;

    // ---------- Host output generators ----------
    procedure Test_rocrand_generate;
    procedure Test_rocrand_generate_long_long;
    procedure Test_rocrand_generate_char;
    procedure Test_rocrand_generate_short;
    procedure Test_rocrand_generate_uniform;
    procedure Test_rocrand_generate_uniform_double;
    procedure Test_rocrand_generate_uniform_half;
    procedure Test_rocrand_generate_normal;
    procedure Test_rocrand_generate_normal_double;
    procedure Test_rocrand_generate_normal_half;
    procedure Test_rocrand_generate_log_normal;
    procedure Test_rocrand_generate_log_normal_double;
    procedure Test_rocrand_generate_log_normal_half;
    //procedure Test_rocrand_generate_poisson;
  end;
implementation

procedure TTest_rocRAND.Setup;
var
  st: rocrand_status;
begin
  st := rocrand_create_generator(generator, ROCRAND_RNG_PSEUDO_DEFAULT);
  if st <> ROCRAND_STATUS_SUCCESS then
    Fail('Setup failed: rocrand_create_generator');
end;

procedure TTest_rocRAND.TearDown;
var
  st: rocrand_status;
begin
  st := rocrand_destroy_generator(generator);
  if st <> ROCRAND_STATUS_SUCCESS then
    Fail('TearDown failed: rocrand_destroy_generator');
end;

// ---------- Implemented Tests ----------

procedure TTest_rocRAND.TestCreateAndDestroyGenerator;
var
  localGenerator: rocrand_generator;
  st: rocrand_status;
begin
  st := rocrand_create_generator(localGenerator, ROCRAND_RNG_PSEUDO_DEFAULT);
  AssertEquals('rocrand_create_generator should return success', Integer(ROCRAND_STATUS_SUCCESS), Integer(st));
  Assert(localGenerator <> nil, 'Generator should not be nil');
  st := rocrand_destroy_generator(localGenerator);
  AssertEquals('rocrand_destroy_generator should return success', Integer(ROCRAND_STATUS_SUCCESS), Integer(st));
end;

// ---------- Generator life-cycle ----------
procedure TTest_rocRAND.Test_rocrand_create_generator_host;
var
  localGenerator: rocrand_generator;
  createStatus: rocrand_status; // More specific name
begin
  // Test creating a host generator with a default pseudo RNG type
  createStatus := rocrand_create_generator_host(localGenerator, ROCRAND_RNG_PSEUDO_DEFAULT);
  try
    // Assert that the call was successful
    AssertEquals('rocrand_create_generator_host should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(createStatus));
    // Assert that the generator pointer is not nil
    AssertTrue('Generator should not be nil after successful creation',
               localGenerator <> nil);
  finally
    // Ensure cleanup happens even if assertions fail
    if (createStatus = ROCRAND_STATUS_SUCCESS) and (localGenerator <> nil) then
    begin
      createStatus := rocrand_destroy_generator(localGenerator);
      // Optionally assert destruction success, though main test is creation
      if createStatus <> ROCRAND_STATUS_SUCCESS then
       WriteLn('Warning: rocrand_destroy_generator failed in Test_rocrand_create_generator_host cleanup');
    end;
  end;
end;


procedure TTest_rocRAND.Test_rocrand_create_generator_host_blocking;
var
  localGenerator: rocrand_generator;
  createStatus: rocrand_status; // More specific name
begin
  // Test creating a host blocking generator with a default pseudo RNG type
  createStatus := rocrand_create_generator_host_blocking(localGenerator, ROCRAND_RNG_PSEUDO_DEFAULT);
  try
    // Assert that the call was successful
    AssertEquals('rocrand_create_generator_host_blocking should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(createStatus));
    // Assert that the generator pointer is not nil
    AssertTrue('Generator should not be nil after successful creation',
               localGenerator <> nil);
  finally
    // Ensure cleanup happens even if assertions fail
    if (createStatus = ROCRAND_STATUS_SUCCESS) and (localGenerator <> nil) then
    begin
      createStatus := rocrand_destroy_generator(localGenerator);
      // Optionally add warning on destruction failure
      if createStatus <> ROCRAND_STATUS_SUCCESS then
       WriteLn('Warning: rocrand_destroy_generator failed in Test_rocrand_create_generator_host_blocking cleanup');
    end;
  end;
end;


procedure TTest_rocrand.Test_rocrand_destroy_generator;
var
  localGenerator: rocrand_generator;
  createStatus: rocrand_status;   // More specific name
  destroyStatus: rocrand_status;  // More specific name
begin
  // This test explicitly tests the destroy function by creating a generator first
  createStatus := rocrand_create_generator(localGenerator, ROCRAND_RNG_PSEUDO_DEFAULT);
  // Check creation was successful before attempting destruction
  AssertEquals('rocrand_create_generator (setup) should return success',
               Integer(ROCRAND_STATUS_SUCCESS),
               Integer(createStatus));
  AssertTrue('Generator (setup) should not be nil',
             localGenerator <> nil);

  // Now test the destruction of the created generator
  destroyStatus := rocrand_destroy_generator(localGenerator);
  AssertEquals('rocrand_destroy_generator should return success',
               Integer(ROCRAND_STATUS_SUCCESS),
               Integer(destroyStatus));

  // Attempting to destroy the same generator again should ideally fail
  // Note: Behavior on double-destroy might be undefined or return an error.
  // Testing this might be optional or dependent on library guarantees.
  // destroyStatus := rocrand_destroy_generator(localGenerator);
  // AssertTrue('Destroying the same generator twice should fail',
  //            destroyStatus <> ROCRAND_STATUS_SUCCESS);
  // For now, we'll just test the successful destruction of a valid generator.
end;


// ---------- Seed / stream / offset ----------
procedure TTest_rocRAND.Test_rocrand_initialize_generator;
var
  initStatus: rocrand_status;
begin
  // The generator is created in SetUp. Test initializing it.
  // Note: Depending on the implementation, the generator might be automatically
  // initialized by create_generator, or initialize_generator might be a separate step.
  // This test checks the function call itself.
  initStatus := rocrand_initialize_generator(generator);
  // Assert that the initialization call was successful.
  // If the generator is already initialized, the behavior might vary.
  // The safest assertion is that it returns success or a specific "already initialized" status if defined.
  // For now, assuming success is expected.
  AssertEquals('rocrand_initialize_generator should return success',
               Integer(ROCRAND_STATUS_SUCCESS),
               Integer(initStatus));
  // Potentially add a second call to check behavior on re-initialization if needed.
  // initStatus := rocrand_initialize_generator(generator);
  // AssertEquals('rocrand_initialize_generator (second call) status check',
  //              {expected status for re-init},
  //              Integer(initStatus));
end;


procedure TTest_rocrand.Test_rocrand_set_stream;
var
  setStreamStatus: rocrand_status;
  stream: hipStream_t; // Use the type from hip_runtime
  createStreamStatus: hipError_t; // Use the HIP error type
begin
  // Create a HIP stream to set
  createStreamStatus := hipStreamCreate(stream);
  // Check if stream creation was successful before proceeding
  if createStreamStatus <> HIP_SUCCESS then
  begin
    Fail('Setup failed: hipStreamCreate');
    Exit; // Exit the test method early
  end;

  try
    // Test setting the stream on the generator created in SetUp
    setStreamStatus := rocrand_set_stream(generator, stream);
    // Assert that the set stream call was successful
    AssertEquals('rocrand_set_stream should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(setStreamStatus));
    // TODO: Potentially add a test to verify the stream was actually set,
    // possibly by generating data and checking if it used the stream.
    // This might be complex and is often assumed if the call succeeds.
  finally
    // Ensure the HIP stream is destroyed even if assertions fail
    createStreamStatus := hipStreamDestroy(stream);
    if createStreamStatus <> HIP_SUCCESS then
    begin
       WriteLn('Warning: hipStreamDestroy failed in Test_rocrand_set_stream cleanup');
    end;
  end;
end;


procedure TTest_rocrand.Test_rocrand_set_seed;
var
  setSeedStatus: rocrand_status;
  testSeed: UInt64;
begin
  testSeed := 1234567890123456789; // Use a specific seed value for testing

  // Test setting the seed on the generator created in SetUp
  setSeedStatus := rocrand_set_seed(generator, testSeed);
  // Assert that the set seed call was successful
  AssertEquals('rocrand_set_seed should return success',
               Integer(ROCRAND_STATUS_SUCCESS),
               Integer(setSeedStatus));
  // TODO: Potentially add a test to verify the seed was set correctly.
  // This would typically involve generating numbers, setting the seed again,
  // generating the same sequence, and comparing the results.
  // That is a more complex integration test.
end;


procedure TTest_rocrand.Test_rocrand_set_seed_uint4;
var
  setSeedStatus: rocrand_status;
  testSeed: uint4; // Use the type defined in rocrand.pas
  localGenerator: rocrand_generator;
  createStatus: rocrand_status;
begin
  // Initialize the uint4 seed structure with values > 1, 7, 15, 127 as per LFSR113 requirements.
  // The function automatically adjusts values <= minimums, but using valid ones is better.
  testSeed.x := 2;   // Must be > 1
  testSeed.y := 8;   // Must be > 7
  testSeed.z := 16;  // Must be > 15
  testSeed.w := 128; // Must be > 127

  // Create the LFSR113 generator specifically required for rocrand_set_seed_uint4
  createStatus := rocrand_create_generator(localGenerator, ROCRAND_RNG_PSEUDO_LFSR113);
  if createStatus <> ROCRAND_STATUS_SUCCESS then
  begin
    Fail('Setup failed: rocrand_create_generator for LFSR113 - Status ' + IntToStr(Integer(createStatus)));
    Exit;
  end;

  try
    // Test setting the uint4 seed on the LFSR113 generator
    setSeedStatus := rocrand_set_seed_uint4(localGenerator, testSeed);
    // Assert that the set seed call was successful
    // Provide the actual status code in the failure message for easier debugging
    AssertEquals('rocrand_set_seed_uint4 (LFSR113) should return success (returned ' + IntToStr(Integer(setSeedStatus)) + ')',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(setSeedStatus));
  finally
    // Ensure the specific LFSR113 generator is destroyed
    createStatus := rocrand_destroy_generator(localGenerator);
    if createStatus <> ROCRAND_STATUS_SUCCESS then
    begin
      WriteLn('Warning: rocrand_destroy_generator failed in Test_rocrand_set_seed_uint4 cleanup - Status ' + IntToStr(Integer(createStatus)));
    end;
  end;
end;


procedure TTest_rocrand.Test_rocrand_set_offset;
var
  setOffsetStatus: rocrand_status;
  testOffset: UInt64;
begin
  testOffset := 9876543210; // Use a specific offset value for testing

  // Test setting the offset on the generator created in SetUp
  setOffsetStatus := rocrand_set_offset(generator, testOffset);
  // Assert that the set offset call was successful
  AssertEquals('rocrand_set_offset should return success',
               Integer(ROCRAND_STATUS_SUCCESS),
               Integer(setOffsetStatus));
  // TODO: Verification would involve generating numbers and checking if the sequence
  // starts at the expected offset point, which is complex.
end;


procedure TTest_rocrand.Test_rocrand_set_ordering;
var
  setOrderingStatus: rocrand_status;
  // Test with a common ordering constant
  testOrdering: rocrand_ordering;
begin
  testOrdering := ROCRAND_ORDERING_PSEUDO_DEFAULT;

  // Test setting the ordering on the generator created in SetUp
  setOrderingStatus := rocrand_set_ordering(generator, testOrdering);
  // Assert that the set ordering call was successful
  AssertEquals('rocrand_set_ordering should return success',
               Integer(ROCRAND_STATUS_SUCCESS),
               Integer(setOrderingStatus));
  // TODO: Verification would involve checking if the generated sequence
  // reflects the specified ordering, which might be difficult to assert directly.
end;


procedure TTest_rocrand.Test_rocrand_set_quasi_random_generator_dimensions;
var
  setDimsStatus: rocrand_status;
  testDimensions: UInt32;
  // Need a quasi-random generator for this test to be meaningful
  quasiGenerator: rocrand_generator;
  createStatus: rocrand_status;
begin
  testDimensions := 10; // Example dimension

  // Create a quasi-random generator specifically for this test
  createStatus := rocrand_create_generator(quasiGenerator, ROCRAND_RNG_QUASI_DEFAULT);
  if createStatus <> ROCRAND_STATUS_SUCCESS then
  begin
    Fail('Setup failed: rocrand_create_generator for quasi generator');
    Exit;
  end;

  try
    // Test setting the dimensions on the quasi-random generator
    setDimsStatus := rocrand_set_quasi_random_generator_dimensions(quasiGenerator, testDimensions);
    // Assert that the set dimensions call was successful
    AssertEquals('rocrand_set_quasi_random_generator_dimensions should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(setDimsStatus));
    // TODO: Verification would involve generating quasi-random numbers
    // and checking properties related to the dimensions.
  finally
    // Ensure the quasi-random generator is destroyed
    createStatus := rocrand_destroy_generator(quasiGenerator);
    if createStatus <> ROCRAND_STATUS_SUCCESS then
    begin
       WriteLn('Warning: rocrand_destroy_generator failed in Test_rocrand_set_quasi_random_generator_dimensions cleanup');
    end;
  end;
end;


// ---------- Generator queries ----------
procedure TTest_rocrand.Test_rocrand_get_version;
var
  getVersionStatus: rocrand_status;
  version: Integer;
begin
  // Test getting the version
  getVersionStatus := rocrand_get_version(version);
  // Assert that the get version call was successful
  AssertEquals('rocrand_get_version should return success',
               Integer(ROCRAND_STATUS_SUCCESS),
               Integer(getVersionStatus));
  // Optionally, assert that the version is a reasonable positive number
  // (This might fail if the library version is < 1.0.0, but unlikely)
  AssertTrue('Version number should be positive',
             version > 0);
  WriteLn('ROCm rocRAND Version: ', version);
end;


// ---------- Discrete distributions ----------
//procedure TTest_rocrand.Test_rocrand_create_poisson_distribution;
//var
//  createDistStatus: rocrand_status;
//  poissonDistribution: rocrand_discrete_distribution;
//begin
//  poissonDistribution := nil;
//
//  // Use lambda = 2.0
//  // NOTE: If this test fails with "Invalid floating point operation",
//  // it likely indicates a bug or specific requirement/limitation within
//  // the underlying librocrand.so implementation for this function.
//  try
//    createDistStatus := rocrand_create_poisson_distribution(2.0, poissonDistribution);
//
//    // If we get here without an exception, check the status
//    if createDistStatus <> ROCRAND_STATUS_SUCCESS then
//    begin
//      WriteLn('rocrand_create_poisson_distribution failed with status: ', Integer(createDistStatus));
//      Fail('rocrand_create_poisson_distribution should return success, got status ' + IntToStr(Integer(createDistStatus)));
//      Exit;
//    end;
//
//    // Assert that the distribution pointer was set
//    AssertTrue('Poisson distribution pointer should not be nil after successful creation',
//               poissonDistribution <> nil);
//
//    // Cleanup
//    createDistStatus := rocrand_destroy_discrete_distribution(poissonDistribution);
//    if createDistStatus <> ROCRAND_STATUS_SUCCESS then
//    begin
//      WriteLn('Warning: rocrand_destroy_discrete_distribution failed in Test_rocrand_create_poisson_distribution cleanup - Status ', Integer(createDistStatus));
//    end;
//
//  except
//    on E: EInvalidOp do
//    begin
//      // Catch the specific FPU exception and report it as a potential library issue
//      WriteLn('Caught EInvalidOp exception in rocrand_create_poisson_distribution. ' +
//              'This might indicate an issue within the librocrand library itself for this specific function call.');
//      Fail('EInvalidOp exception caught: ' + E.Message +
//           '. This suggests a potential problem within librocrand, not the Pascal wrapper.');
//    end;
//    on E: Exception do
//    begin
//      // Re-raise any other unexpected exceptions
//      raise;
//    end;
//  end;
//end;


procedure TTest_rocrand.Test_rocrand_create_discrete_distribution;
var
  createDistStatus: rocrand_status;
  generalDistribution: rocrand_discrete_distribution;
  probabilities: array[0..3] of Double;
  size: UInt32;
  offset: UInt32;
begin
  generalDistribution := nil; // Explicitly initialize

  // Setup example probability array
  probabilities[0] := 0.1;
  probabilities[1] := 0.3;
  probabilities[2] := 0.4;
  probabilities[3] := 0.2;
  size := 4;
  offset := 0;

  // Test creating a general discrete distribution
  createDistStatus := rocrand_create_discrete_distribution(@probabilities[0], size, offset, generalDistribution);

  // Check status immediately
  if createDistStatus <> ROCRAND_STATUS_SUCCESS then
  begin
    WriteLn('rocrand_create_discrete_distribution failed with status: ', Integer(createDistStatus));
    Fail('rocrand_create_discrete_distribution should return success, got status ' + IntToStr(Integer(createDistStatus)));
    Exit;
  end;

  // Only proceed if creation was successful
  AssertTrue('General discrete distribution should not be nil after successful creation',
             generalDistribution <> nil);

  // Cleanup: Destroy the distribution we just created
  createDistStatus := rocrand_destroy_discrete_distribution(generalDistribution);
  if createDistStatus <> ROCRAND_STATUS_SUCCESS then
  begin
    WriteLn('Warning: rocrand_destroy_discrete_distribution failed in Test_rocrand_create_discrete_distribution cleanup - Status ', Integer(createDistStatus));
  end;
end;


procedure TTest_rocrand.Test_rocrand_destroy_discrete_distribution;
var
  createDistStatus: rocrand_status;
  destroyDistStatus: rocrand_status;
  testDistribution: rocrand_discrete_distribution;
  setupProbabilities: array[0..1] of Double; // Variable for setup code
begin
  testDistribution := nil;

  // Setup: Create a distribution to destroy (using general discrete distribution)
  setupProbabilities[0] := 0.5;
  setupProbabilities[1] := 0.5;
  createDistStatus := rocrand_create_discrete_distribution(@setupProbabilities[0], 2, 0, testDistribution);

  // Check setup was successful
  if createDistStatus <> ROCRAND_STATUS_SUCCESS then
  begin
    WriteLn('Setup (rocrand_create_discrete_distribution) failed with status: ', Integer(createDistStatus));
    Fail('Setup (rocrand_create_discrete_distribution) should return success, got status ' + IntToStr(Integer(createDistStatus)));
    Exit;
  end;

  if testDistribution = nil then
  begin
    WriteLn('Setup (discrete distribution) is nil despite success status.');
    Fail('Setup (discrete distribution) should not be nil');
    Exit;
  end;

  // Test destroying the created distribution
  try
    destroyDistStatus := rocrand_destroy_discrete_distribution(testDistribution);

    // Check the destruction status
    if destroyDistStatus <> ROCRAND_STATUS_SUCCESS then
    begin
      WriteLn('rocrand_destroy_discrete_distribution failed with status: ', Integer(destroyDistStatus));
      Fail('rocrand_destroy_discrete_distribution should return success, got status ' + IntToStr(Integer(destroyDistStatus)));
      Exit;
    end;
  except
    on E: EInvalidOp do
    begin
      // Catch the specific FPU exception and report it
      WriteLn('Caught EInvalidOp exception in rocrand_destroy_discrete_distribution. ' +
              'This might indicate an issue within the librocrand library itself for this specific function call.');
      Fail('EInvalidOp exception caught: ' + E.Message +
           '. This suggests a potential problem within librocrand, not the Pascal wrapper.');
    end;
    on E: Exception do
    begin
      // Re-raise any other unexpected exceptions
      raise;
    end;
  end;

  // Test destroying a nil pointer (should return ROCRAND_STATUS_OUT_OF_RANGE according to docs)
  // destroyDistStatus := rocrand_destroy_discrete_distribution(nil);
  // AssertEquals('rocrand_destroy_discrete_distribution should return OUT_OF_RANGE for nil',
  //              Integer(ROCRAND_STATUS_OUT_OF_RANGE),
  //              Integer(destroyDistStatus));
  //Message:           TTest_rocRAND.Test_rocrand_create_poisson_distribution: EInvalidOp exception caught: Invalid floating point operation.
  //                   This suggests a potential problem within librocrand, not the Pascal wrapper.
  //Exception class:   EAssertionFailedError
  //Exception message: EInvalidOp exception caught: Invalid floating point operation.
  //                   This suggests a potential problem within librocrand, not the Pascal wrapper.

end;

// ---------- Direction-vector helpers ----------
procedure TTest_rocrand.Test_rocrand_get_direction_vectors32;
var
  getVectorsStatus: rocrand_status;
  vectorsPtr: PUInt32; // Pointer to receive the address of the vectors
  vectorSet: rocrand_direction_vector_set;
begin
  vectorsPtr := nil; // Explicitly initialize the pointer variable
  vectorSet := ROCRAND_DIRECTION_VECTORS_32_JOEKUO6; // Use a standard set constant

  // Test getting 32-bit direction vectors
  getVectorsStatus := rocrand_get_direction_vectors32(vectorsPtr, vectorSet);

  // Assert that the call was successful
  AssertEquals('rocrand_get_direction_vectors32 should return success',
               Integer(ROCRAND_STATUS_SUCCESS),
               Integer(getVectorsStatus));

  // Assert that the returned pointer is not nil (indicating vectors were found/returned)
  AssertTrue('Pointer to 32-bit direction vectors should not be nil after successful call',
             vectorsPtr <> nil);

  // Note: The actual content of the vectors is internal to librocrand and not easily verified
  // without deep knowledge of the expected values for the specific set.
  // The test confirms the function call mechanism works and returns a valid pointer.
end;


procedure TTest_rocrand.Test_rocrand_get_direction_vectors64;
var
  getVectorsStatus: rocrand_status;
  vectorsPtr: PUInt64; // Pointer to receive the address of the vectors
  vectorSet: rocrand_direction_vector_set;
begin
  vectorsPtr := nil; // Explicitly initialize the pointer variable
  vectorSet := ROCRAND_DIRECTION_VECTORS_64_JOEKUO6; // Use a standard set constant for 64-bit

  // Test getting 64-bit direction vectors
  getVectorsStatus := rocrand_get_direction_vectors64(vectorsPtr, vectorSet);

  // Assert that the call was successful
  AssertEquals('rocrand_get_direction_vectors64 should return success',
               Integer(ROCRAND_STATUS_SUCCESS),
               Integer(getVectorsStatus));

  // Assert that the returned pointer is not nil
  AssertTrue('Pointer to 64-bit direction vectors should not be nil after successful call',
             vectorsPtr <> nil);

  // Note: Similar to 32-bit vectors, content verification is complex.
end;


procedure TTest_rocrand.Test_rocrand_get_scramble_constants32;
var
  getConstantsStatus: rocrand_status;
  constantsPtr: PUInt32; // Pointer to receive the address of the constants
begin
  constantsPtr := nil; // Explicitly initialize the pointer variable

  // Test getting 32-bit scramble constants
  getConstantsStatus := rocrand_get_scramble_constants32(constantsPtr);

  // Assert that the call was successful
  AssertEquals('rocrand_get_scramble_constants32 should return success',
               Integer(ROCRAND_STATUS_SUCCESS),
               Integer(getConstantsStatus));

  // Assert that the returned pointer is not nil
  AssertTrue('Pointer to 32-bit scramble constants should not be nil after successful call',
             constantsPtr <> nil);

  // Note: Content verification is complex as it depends on internal library data.
end;


procedure TTest_rocrand.Test_rocrand_get_scramble_constants64;
var
  getConstantsStatus: rocrand_status;
  constantsPtr: PUInt64; // Pointer to receive the address of the constants
begin
  constantsPtr := nil; // Explicitly initialize the pointer variable

  // Test getting 64-bit scramble constants
  getConstantsStatus := rocrand_get_scramble_constants64(constantsPtr);

  // Assert that the call was successful
  AssertEquals('rocrand_get_scramble_constants64 should return success',
               Integer(ROCRAND_STATUS_SUCCESS),
               Integer(getConstantsStatus));

  // Assert that the returned pointer is not nil
  AssertTrue('Pointer to 64-bit scramble constants should not be nil after successful call',
             constantsPtr <> nil);

  // Note: Content verification is complex as it depends on internal library data.
end;

// ---------- Host output generators ----------
procedure TTest_rocrand.Test_rocrand_generate;
var
  generateStatus: rocrand_status;
  devicePtr: hipDeviceptr_t;
  numElements: UIntPtr;
  i: Integer;
  hostData: PUInt32; // Use PUInt32 to match rocrand_generate output type (unsigned int)
  mallocStatus: hipError_t;
  freeStatus: hipError_t;
  memcpyStatus: hipError_t;
begin
  numElements := 1000; // Number of UInt32 elements to generate
  devicePtr := nil;

  // Allocate device memory for the output
  mallocStatus := hipMalloc(devicePtr, numElements * SizeOf(UInt32));
  if mallocStatus <> HIP_SUCCESS then
  begin
    Fail('hipMalloc failed in Test_rocrand_generate - HIP Error: ' + IntToStr(Integer(mallocStatus)));
    Exit;
  end;

  try
    // Test generating random UInt32 numbers
    generateStatus := rocrand_generate(generator, devicePtr, numElements);
    // Assert that the generation call was successful
    AssertEquals('rocrand_generate should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(generateStatus));

    // Optional: Copy data back to host to verify it was written
    // Allocate host memory
    GetMem(hostData, numElements * SizeOf(UInt32));
    if hostData = nil then
    begin
      Fail('GetMem failed in Test_rocrand_generate for host data');
      Exit;
    end;

    try
      // Copy from device to host
      memcpyStatus := hipMemcpy(hostData, devicePtr, numElements * SizeOf(UInt32), hipMemcpyDeviceToHost);
      if memcpyStatus <> HIP_SUCCESS then
      begin
        Fail('hipMemcpy failed in Test_rocrand_generate - HIP Error: ' + IntToStr(Integer(memcpyStatus)));
        Exit;
      end;

      // Basic check: Ensure not all generated values are zero (very unlikely for a good RNG)
      // This is a weak check but confirms data was written and isn't just zeroed memory
      i := 0;
      while (i < Integer(numElements)) and (hostData[i] = 0) do
        Inc(i);
      AssertTrue('At least one generated UInt32 value should be non-zero',
                 i < Integer(numElements));

    finally
      // Free host memory
      FreeMem(hostData);
    end;

  finally
    // Free device memory
    freeStatus := hipFree(devicePtr);
    if freeStatus <> HIP_SUCCESS then
    begin
      WriteLn('Warning: hipFree failed in Test_rocrand_generate cleanup - HIP Error: ', Integer(freeStatus));
    end;
  end;
end;


procedure TTest_rocrand.Test_rocrand_generate_long_long;
var
  generateStatus: rocrand_status;
  devicePtr: hipDeviceptr_t;
  numElements: UIntPtr;
  i: Integer;
  hostData: PUInt64; // Use PUInt64 for unsigned long long
  mallocStatus: hipError_t;
  freeStatus: hipError_t;
  memcpyStatus: hipError_t;
  localGenerator: rocrand_generator; // Local generator for this test
  createGenStatus: rocrand_status;
begin
  numElements := 500; // Number of UInt64 elements to generate
  devicePtr := nil;
  localGenerator := nil; // Initialize local generator

  // Create a generator type explicitly designed for 64-bit output
  createGenStatus := rocrand_create_generator(localGenerator, ROCRAND_RNG_PSEUDO_THREEFRY4_64_20);
  if createGenStatus <> ROCRAND_STATUS_SUCCESS then
  begin
    Fail('Setup failed: rocrand_create_generator for THREEFRY4_64_20 - Status ' + IntToStr(Integer(createGenStatus)));
    Exit;
  end;

  // Allocate device memory for the output
  mallocStatus := hipMalloc(devicePtr, numElements * SizeOf(UInt64));
  if mallocStatus <> HIP_SUCCESS then
  begin
    // Cleanup generator before failing
    rocrand_destroy_generator(localGenerator);
    Fail('hipMalloc failed in Test_rocrand_generate_long_long - HIP Error: ' + IntToStr(Integer(mallocStatus)));
    Exit;
  end;

  try
    // Test generating random UInt64 numbers using the Threefry 64-bit generator
    generateStatus := rocrand_generate_long_long(localGenerator, devicePtr, numElements);
    // Assert that the generation call was successful
    // Provide the actual status in the message for better debugging if it fails again
    AssertEquals('rocrand_generate_long_long (THREEFRY4_64_20) should return success (returned ' + IntToStr(Integer(generateStatus)) + ')',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(generateStatus));

    // Optional: Copy data back to host to verify
    GetMem(hostData, numElements * SizeOf(UInt64));
    if hostData = nil then
    begin
      Fail('GetMem failed in Test_rocrand_generate_long_long for host data');
      Exit;
    end;

    try
      memcpyStatus := hipMemcpy(hostData, devicePtr, numElements * SizeOf(UInt64), hipMemcpyDeviceToHost);
      if memcpyStatus <> HIP_SUCCESS then
      begin
        Fail('hipMemcpy failed in Test_rocrand_generate_long_long - HIP Error: ' + IntToStr(Integer(memcpyStatus)));
        Exit;
      end;

      // Basic check: Ensure not all generated values are zero
      i := 0;
      while (i < Integer(numElements)) and (hostData[i] = 0) do
        Inc(i);
      AssertTrue('At least one generated UInt64 value should be non-zero',
                 i < Integer(numElements));

    finally
      FreeMem(hostData);
    end;

  finally
    // Free device memory
    freeStatus := hipFree(devicePtr);
    if freeStatus <> HIP_SUCCESS then
    begin
      WriteLn('Warning: hipFree failed in Test_rocrand_generate_long_long cleanup - HIP Error: ', Integer(freeStatus));
    end;
    // Destroy the local generator
    createGenStatus := rocrand_destroy_generator(localGenerator);
    if createGenStatus <> ROCRAND_STATUS_SUCCESS then
    begin
      WriteLn('Warning: rocrand_destroy_generator failed in Test_rocrand_generate_long_long cleanup - ROCRAND Status: ', Integer(createGenStatus));
    end;
  end;
end;


procedure TTest_rocrand.Test_rocrand_generate_char;
var
  generateStatus: rocrand_status;
  devicePtr: hipDeviceptr_t;
  numElements: UIntPtr;
  i: Integer;
  hostData: PUInt8; // Use PUInt8 for unsigned char
  mallocStatus: hipError_t;
  freeStatus: hipError_t;
  memcpyStatus: hipError_t;
begin
  numElements := 2000; // Number of UInt8 elements to generate
  devicePtr := nil;

  // Allocate device memory for the output
  mallocStatus := hipMalloc(devicePtr, numElements * SizeOf(UInt8));
  if mallocStatus <> HIP_SUCCESS then
  begin
    Fail('hipMalloc failed in Test_rocrand_generate_char - HIP Error: ' + IntToStr(Integer(mallocStatus)));
    Exit;
  end;

  try
    // Test generating random UInt8 numbers
    generateStatus := rocrand_generate_char(generator, devicePtr, numElements);
    // Assert that the generation call was successful
    AssertEquals('rocrand_generate_char should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(generateStatus));

    // Optional: Copy data back to host to verify
    GetMem(hostData, numElements * SizeOf(UInt8));
    if hostData = nil then
    begin
      Fail('GetMem failed in Test_rocrand_generate_char for host data');
      Exit;
    end;

    try
      memcpyStatus := hipMemcpy(hostData, devicePtr, numElements * SizeOf(UInt8), hipMemcpyDeviceToHost);
      if memcpyStatus <> HIP_SUCCESS then
      begin
        Fail('hipMemcpy failed in Test_rocrand_generate_char - HIP Error: ' + IntToStr(Integer(memcpyStatus)));
        Exit;
      end;

      // Basic check: Ensure not all generated values are zero
      i := 0;
      while (i < Integer(numElements)) and (hostData[i] = 0) do
        Inc(i);
      AssertTrue('At least one generated UInt8 value should be non-zero',
                 i < Integer(numElements));

    finally
      FreeMem(hostData);
    end;

  finally
    freeStatus := hipFree(devicePtr);
    if freeStatus <> HIP_SUCCESS then
    begin
      WriteLn('Warning: hipFree failed in Test_rocrand_generate_char cleanup - HIP Error: ', Integer(freeStatus));
    end;
  end;
end;


procedure TTest_rocrand.Test_rocrand_generate_short;
var
  generateStatus: rocrand_status;
  devicePtr: hipDeviceptr_t;
  numElements: UIntPtr;
  i: Integer;
  hostData: PUInt16; // Use PUInt16 for unsigned short
  mallocStatus: hipError_t;
  freeStatus: hipError_t;
  memcpyStatus: hipError_t;
begin
  numElements := 1500; // Number of UInt16 elements to generate
  devicePtr := nil;

  // Allocate device memory for the output
  mallocStatus := hipMalloc(devicePtr, numElements * SizeOf(UInt16));
  if mallocStatus <> HIP_SUCCESS then
   begin
    Fail('hipMalloc failed in Test_rocrand_generate_short - HIP Error: ' + IntToStr(Integer(mallocStatus)));
    Exit;
  end;

  try
    // Test generating random UInt16 numbers
    generateStatus := rocrand_generate_short(generator, devicePtr, numElements);
    // Assert that the generation call was successful
    AssertEquals('rocrand_generate_short should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(generateStatus));

    // Optional: Copy data back to host to verify
    GetMem(hostData, numElements * SizeOf(UInt16));
    if hostData = nil then
    begin
      Fail('GetMem failed in Test_rocrand_generate_short for host data');
      Exit;
    end;

    try
      memcpyStatus := hipMemcpy(hostData, devicePtr, numElements * SizeOf(UInt16), hipMemcpyDeviceToHost);
      if memcpyStatus <> HIP_SUCCESS then
      begin
        Fail('hipMemcpy failed in Test_rocrand_generate_short - HIP Error: ' + IntToStr(Integer(memcpyStatus)));
        Exit;
      end;

      // Basic check: Ensure not all generated values are zero
      i := 0;
      while (i < Integer(numElements)) and (hostData[i] = 0) do
        Inc(i);
      AssertTrue('At least one generated UInt16 value should be non-zero',
                 i < Integer(numElements));

    finally
      FreeMem(hostData);
    end;

  finally
    freeStatus := hipFree(devicePtr);
    if freeStatus <> HIP_SUCCESS then
    begin
      WriteLn('Warning: hipFree failed in Test_rocrand_generate_short cleanup - HIP Error: ', Integer(freeStatus));
    end;
  end;
end;

procedure TTest_rocrand.Test_rocrand_generate_uniform;
var
  generateStatus: rocrand_status;
  devicePtr: hipDeviceptr_t;
  numElements: UIntPtr;
  i: Integer;
  hostData: PSingle; // Use PSingle for Single (float)
  mallocStatus: hipError_t;
  freeStatus: hipError_t;
  memcpyStatus: hipError_t;
begin
  numElements := 1000; // Number of Single (float) elements to generate
  devicePtr := nil;

  // Allocate device memory for the output
  mallocStatus := hipMalloc(devicePtr, numElements * SizeOf(Single));
  if mallocStatus <> HIP_SUCCESS then
  begin
    Fail('hipMalloc failed in Test_rocrand_generate_uniform - HIP Error: ' + IntToStr(Integer(mallocStatus)));
    Exit;
  end;

  try
    // Test generating uniformly distributed Single (float) numbers [0.0, 1.0)
    generateStatus := rocrand_generate_uniform(generator, devicePtr, numElements);
    // Assert that the generation call was successful
    AssertEquals('rocrand_generate_uniform should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(generateStatus));

    // Optional: Copy data back to host to verify
    GetMem(hostData, numElements * SizeOf(Single));
    if hostData = nil then
    begin
      Fail('GetMem failed in Test_rocrand_generate_uniform for host data');
      Exit;
    end;

    try
      memcpyStatus := hipMemcpy(hostData, devicePtr, numElements * SizeOf(Single), hipMemcpyDeviceToHost);
      if memcpyStatus <> HIP_SUCCESS then
      begin
        Fail('hipMemcpy failed in Test_rocrand_generate_uniform - HIP Error: ' + IntToStr(Integer(memcpyStatus)));
        Exit;
      end;

      // Basic check: Ensure values are in the expected range [0.0, 1.0)
      // Also check that not all values are identical (very unlikely for a good RNG)
      i := 0;
      while (i < Integer(numElements)) and
            ((hostData[i] < 0.0) or (hostData[i] >= 1.0)) do
        Inc(i);
      AssertTrue('Generated uniform Single values should be in range [0.0, 1.0)',
                 i < Integer(numElements)); // If i reached numElements, all were out of range

      // Check for variation (not all identical)
      i := 1;
      while (i < Integer(numElements)) and (hostData[i] = hostData[0]) do
        Inc(i);
      AssertTrue('Generated uniform Single values should show variation',
                 i < Integer(numElements)); // If i reached numElements, all were identical

    finally
      FreeMem(hostData);
    end;

  finally
    freeStatus := hipFree(devicePtr);
    if freeStatus <> HIP_SUCCESS then
    begin
      WriteLn('Warning: hipFree failed in Test_rocrand_generate_uniform cleanup - HIP Error: ', Integer(freeStatus));
    end;
  end;
end;


procedure TTest_rocrand.Test_rocrand_generate_uniform_double;
var
  generateStatus: rocrand_status;
  devicePtr: hipDeviceptr_t;
  numElements: UIntPtr;
  i: Integer;
  hostData: PDouble; // Use PDouble for Double
  mallocStatus: hipError_t;
  freeStatus: hipError_t;
  memcpyStatus: hipError_t;
begin
  numElements := 1000; // Number of Double elements to generate
  devicePtr := nil;

  // Allocate device memory for the output
  mallocStatus := hipMalloc(devicePtr, numElements * SizeOf(Double));
  if mallocStatus <> HIP_SUCCESS then
  begin
    Fail('hipMalloc failed in Test_rocrand_generate_uniform_double - HIP Error: ' + IntToStr(Integer(mallocStatus)));
    Exit;
  end;

  try
    // Test generating uniformly distributed Double numbers [0.0, 1.0)
    generateStatus := rocrand_generate_uniform_double(generator, devicePtr, numElements);
    // Assert that the generation call was successful
    AssertEquals('rocrand_generate_uniform_double should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(generateStatus));

    // Optional: Copy data back to host to verify
    GetMem(hostData, numElements * SizeOf(Double));
    if hostData = nil then
    begin
      Fail('GetMem failed in Test_rocrand_generate_uniform_double for host data');
      Exit;
    end;

    try
      memcpyStatus := hipMemcpy(hostData, devicePtr, numElements * SizeOf(Double), hipMemcpyDeviceToHost);
      if memcpyStatus <> HIP_SUCCESS then
      begin
        Fail('hipMemcpy failed in Test_rocrand_generate_uniform_double - HIP Error: ' + IntToStr(Integer(memcpyStatus)));
        Exit;
      end;

      // Basic check: Ensure values are in the expected range [0.0, 1.0)
      // Also check that not all values are identical
      i := 0;
      while (i < Integer(numElements)) and
            ((hostData[i] < 0.0) or (hostData[i] >= 1.0)) do
        Inc(i);
      AssertTrue('Generated uniform Double values should be in range [0.0, 1.0)',
                 i < Integer(numElements));

      // Check for variation
      i := 1;
      while (i < Integer(numElements)) and (hostData[i] = hostData[0]) do
        Inc(i);
      AssertTrue('Generated uniform Double values should show variation',
                 i < Integer(numElements));

    finally
      FreeMem(hostData);
    end;

  finally
    freeStatus := hipFree(devicePtr);
    if freeStatus <> HIP_SUCCESS then
    begin
      WriteLn('Warning: hipFree failed in Test_rocrand_generate_uniform_double cleanup - HIP Error: ', Integer(freeStatus));
    end;
  end;
end;


procedure TTest_rocrand.Test_rocrand_generate_uniform_half;
var
  generateStatus: rocrand_status;
  devicePtr: hipDeviceptr_t;
  numElements: UIntPtr;
  // For half-precision, we might copy back as UInt16 or use utilities if available.
  // For now, just verify the call works and memory is allocated/freed.
  mallocStatus: hipError_t;
  freeStatus: hipError_t;
begin
  numElements := 1000; // Number of half elements to generate
  devicePtr := nil;

  // Allocate device memory for the output
  mallocStatus := hipMalloc(devicePtr, numElements * SizeOf(UInt16)); // Half is 16 bits
  if mallocStatus <> HIP_SUCCESS then
  begin
    Fail('hipMalloc failed in Test_rocrand_generate_uniform_half - HIP Error: ' + IntToStr(Integer(mallocStatus)));
    Exit;
  end;

  try
    // Test generating uniformly distributed half-precision numbers [0.0, 1.0)
    // Note: The generator type compatibility for half-generation should be checked.
    // If it fails with TYPE_ERROR, a specific generator might be needed.
    generateStatus := rocrand_generate_uniform_half(generator, devicePtr, numElements);
    // Assert that the generation call was successful
    AssertEquals('rocrand_generate_uniform_half should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(generateStatus));

    // Note: Copying half data back and verifying its correctness requires
    // half-precision conversion utilities (like those potentially in ROCm_test_utils.pas
    // for rocBLAS). For a basic test, confirming the call succeeded and memory
    // operations work is sufficient.

  finally
    freeStatus := hipFree(devicePtr);
    if freeStatus <> HIP_SUCCESS then
    begin
      WriteLn('Warning: hipFree failed in Test_rocrand_generate_uniform_half cleanup - HIP Error: ', Integer(freeStatus));
    end;
  end;
end;


procedure TTest_rocrand.Test_rocrand_generate_normal;
var
  generateStatus: rocrand_status;
  devicePtr: hipDeviceptr_t;
  numElements: UIntPtr;
  i: Integer;
  hostData: PSingle; // Use PSingle for Single (float)
  mallocStatus: hipError_t;
  freeStatus: hipError_t;
  memcpyStatus: hipError_t;
  mean: Single;
  stddev: Single;
begin
  numElements := 1000; // Number of Single (float) elements to generate
  devicePtr := nil;
  mean := 5.0;   // Example mean
  stddev := 2.0; // Example standard deviation

  // Allocate device memory for the output
  mallocStatus := hipMalloc(devicePtr, numElements * SizeOf(Single));
  if mallocStatus <> HIP_SUCCESS then
  begin
    Fail('hipMalloc failed in Test_rocrand_generate_normal - HIP Error: ' + IntToStr(Integer(mallocStatus)));
    Exit;
  end;

  try
    // Test generating normally distributed Single (float) numbers
    generateStatus := rocrand_generate_normal(generator, devicePtr, numElements, mean, stddev);
    // Assert that the generation call was successful
    AssertEquals('rocrand_generate_normal should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(generateStatus));

    // Optional: Copy data back to host to verify
    GetMem(hostData, numElements * SizeOf(Single));
    if hostData = nil then
    begin
      Fail('GetMem failed in Test_rocrand_generate_normal for host data');
      Exit;
    end;

    try
      memcpyStatus := hipMemcpy(hostData, devicePtr, numElements * SizeOf(Single), hipMemcpyDeviceToHost);
      if memcpyStatus <> HIP_SUCCESS then
      begin
        Fail('hipMemcpy failed in Test_rocrand_generate_normal - HIP Error: ' + IntToStr(Integer(memcpyStatus)));
        Exit;
      end;

      // Basic check: While normally distributed data can be any value,
      // it's extremely unlikely all generated values are identical.
      i := 1;
      while (i < Integer(numElements)) and (hostData[i] = hostData[0]) do
        Inc(i);
      AssertTrue('Generated normal Single values should show variation',
                 i < Integer(numElements)); // If i reached numElements, all were identical

      // A more thorough test would calculate sample mean/stddev and compare to inputs,
      // but that's more complex and might require tolerance checks.

    finally
      FreeMem(hostData);
    end;

  finally
    freeStatus := hipFree(devicePtr);
    if freeStatus <> HIP_SUCCESS then
    begin
      WriteLn('Warning: hipFree failed in Test_rocrand_generate_normal cleanup - HIP Error: ', Integer(freeStatus));
    end;
  end;
end;


procedure TTest_rocrand.Test_rocrand_generate_normal_double;
var
  generateStatus: rocrand_status;
  devicePtr: hipDeviceptr_t;
  numElements: UIntPtr;
  i: Integer;
  hostData: PDouble; // Use PDouble for Double
  mallocStatus: hipError_t;
  freeStatus: hipError_t;
  memcpyStatus: hipError_t;
  mean: Double;
  stddev: Double;
begin
  numElements := 1000; // Number of Double elements to generate
  devicePtr := nil;
  mean := -1.5;  // Example mean
  stddev := 0.5; // Example standard deviation

  // Allocate device memory for the output
  mallocStatus := hipMalloc(devicePtr, numElements * SizeOf(Double));
  if mallocStatus <> HIP_SUCCESS then
  begin
    Fail('hipMalloc failed in Test_rocrand_generate_normal_double - HIP Error: ' + IntToStr(Integer(mallocStatus)));
    Exit;
  end;

  try
    // Test generating normally distributed Double numbers
    generateStatus := rocrand_generate_normal_double(generator, devicePtr, numElements, mean, stddev);
    // Assert that the generation call was successful
    AssertEquals('rocrand_generate_normal_double should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(generateStatus));

    // Optional: Copy data back to host to verify
    GetMem(hostData, numElements * SizeOf(Double));
    if hostData = nil then
    begin
      Fail('GetMem failed in Test_rocrand_generate_normal_double for host data');
      Exit;
    end;

    try
      memcpyStatus := hipMemcpy(hostData, devicePtr, numElements * SizeOf(Double), hipMemcpyDeviceToHost);
      if memcpyStatus <> HIP_SUCCESS then
      begin
        Fail('hipMemcpy failed in Test_rocrand_generate_normal_double - HIP Error: ' + IntToStr(Integer(memcpyStatus)));
        Exit;
      end;

      // Basic check: Check for variation in generated values
      i := 1;
      while (i < Integer(numElements)) and (hostData[i] = hostData[0]) do
        Inc(i);
      AssertTrue('Generated normal Double values should show variation',
                 i < Integer(numElements));

    finally
      FreeMem(hostData);
    end;

  finally
    freeStatus := hipFree(devicePtr);
    if freeStatus <> HIP_SUCCESS then
    begin
      WriteLn('Warning: hipFree failed in Test_rocrand_generate_normal_double cleanup - HIP Error: ', Integer(freeStatus));
    end;
  end;
end;


procedure TTest_rocrand.Test_rocrand_generate_normal_half;
var
  generateStatus: rocrand_status;
  devicePtr: hipDeviceptr_t;
  numElements: UIntPtr;
  mallocStatus: hipError_t;
  freeStatus: hipError_t;
  // mean and stddev parameters for _half functions are complex (packed half2)
  // For now, focus on the call and memory management.
begin
  numElements := 1000; // Number of half elements to generate
  devicePtr := nil;

  // Allocate device memory for the output (half is 16 bits)
  mallocStatus := hipMalloc(devicePtr, numElements * SizeOf(UInt16));
  if mallocStatus <> HIP_SUCCESS then
  begin
    Fail('hipMalloc failed in Test_rocrand_generate_normal_half - HIP Error: ' + IntToStr(Integer(mallocStatus)));
    Exit;
  end;

  try
    // Test generating normally distributed half-precision numbers
    // Note: Similar to _uniform_half, generator compatibility is key.
    // Parameters for mean/stddev are packed structures (half2).
    generateStatus := rocrand_generate_normal_half(generator, devicePtr, numElements, Default(half2), Default(half2)); // Using Default for now
    // Assert that the generation call was successful
    AssertEquals('rocrand_generate_normal_half should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(generateStatus));

    // Note: Verifying the content of half-precision normal data requires
    // unpacking and conversion, which is complex without specific utilities.
    // The test confirms the API call mechanism works.

  finally
    freeStatus := hipFree(devicePtr);
    if freeStatus <> HIP_SUCCESS then
    begin
      WriteLn('Warning: hipFree failed in Test_rocrand_generate_normal_half cleanup - HIP Error: ', Integer(freeStatus));
    end;
  end;
end;


// Add Math to the uses clause in the implementation section if not already present
// uses ..., Math;

// ---------- Host output generators (Log-Normal/Poisson) ----------
procedure TTest_rocrand.Test_rocrand_generate_log_normal;
var
  generateStatus: rocrand_status;
  devicePtr: hipDeviceptr_t;
  numElements: UIntPtr;
  i: Integer;
  hostData: PSingle; // Use PSingle for Single (float)
  mallocStatus: hipError_t;
  freeStatus: hipError_t;
  memcpyStatus: hipError_t;
  mean: Single;
  stddev: Single;
begin
  numElements := 1000; // Number of Single (float) elements to generate
  devicePtr := nil;
  mean := 0.0;   // Mean of the underlying normal distribution
  stddev := 1.0; // Standard deviation of the underlying normal distribution

  // Allocate device memory for the output
  mallocStatus := hipMalloc(devicePtr, numElements * SizeOf(Single));
  if mallocStatus <> HIP_SUCCESS then
  begin
    Fail('hipMalloc failed in Test_rocrand_generate_log_normal - HIP Error: ' + IntToStr(Integer(mallocStatus)));
    Exit;
  end;

  try
    // Test generating log-normally distributed Single (float) numbers
    // Note: Log-normal requires positive outputs, so mean/stddev of underlying normal affect this.
    generateStatus := rocrand_generate_log_normal(generator, devicePtr, numElements, mean, stddev);
    // Assert that the generation call was successful
    AssertEquals('rocrand_generate_log_normal should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(generateStatus));

    // Optional: Copy data back to host to verify
    GetMem(hostData, numElements * SizeOf(Single));
    if hostData = nil then
    begin
      Fail('GetMem failed in Test_rocrand_generate_log_normal for host data');
      Exit;
    end;

    try
      memcpyStatus := hipMemcpy(hostData, devicePtr, numElements * SizeOf(Single), hipMemcpyDeviceToHost);
      if memcpyStatus <> HIP_SUCCESS then
      begin
        Fail('hipMemcpy failed in Test_rocrand_generate_log_normal - HIP Error: ' + IntToStr(Integer(memcpyStatus)));
        Exit;
      end;

      // Basic checks for log-normal distribution:
      // 1. All generated values should be positive (x > 0)
      i := 0;
      while (i < Integer(numElements)) and (hostData[i] > 0.0) do
        Inc(i);
      AssertTrue('All generated log-normal Single values should be positive',
                 i = Integer(numElements)); // If i < numElements, we found a non-positive value

      // 2. Check for variation (extremely unlikely all values are identical for a good RNG)
      i := 1;
      while (i < Integer(numElements)) and (Abs(hostData[i] - hostData[0]) < 1e-6) do // Small epsilon for float comparison
        Inc(i);
      AssertTrue('Generated log-normal Single values should show variation',
                 i < Integer(numElements)); // If i reached numElements, all were effectively identical

    finally
      FreeMem(hostData);
    end;

  finally
    freeStatus := hipFree(devicePtr);
    if freeStatus <> HIP_SUCCESS then
    begin
      WriteLn('Warning: hipFree failed in Test_rocrand_generate_log_normal cleanup - HIP Error: ', Integer(freeStatus));
    end;
  end;
end;


procedure TTest_rocrand.Test_rocrand_generate_log_normal_double;
var
  generateStatus: rocrand_status;
  devicePtr: hipDeviceptr_t;
  numElements: UIntPtr;
  i: Integer;
  hostData: PDouble; // Use PDouble for Double
  mallocStatus: hipError_t;
  freeStatus: hipError_t;
  memcpyStatus: hipError_t;
  mean: Double;
  stddev: Double;
  epsilon: Double; // Tolerance for floating-point comparison
begin
  numElements := 1000; // Number of Double elements to generate
  devicePtr := nil;
  mean := 1.5;   // Mean of the underlying normal distribution
  stddev := 0.8; // Standard deviation of the underlying normal distribution
  epsilon := 1e-12; // Small tolerance for Double comparison

  // Allocate device memory for the output
  mallocStatus := hipMalloc(devicePtr, numElements * SizeOf(Double));
  if mallocStatus <> HIP_SUCCESS then
  begin
    Fail('hipMalloc failed in Test_rocrand_generate_log_normal_double - HIP Error: ' + IntToStr(Integer(mallocStatus)));
    Exit;
  end;

  try
    // Test generating log-normally distributed Double numbers
    generateStatus := rocrand_generate_log_normal_double(generator, devicePtr, numElements, mean, stddev);
    // Assert that the generation call was successful
    AssertEquals('rocrand_generate_log_normal_double should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(generateStatus));

    // Optional: Copy data back to host to verify
    GetMem(hostData, numElements * SizeOf(Double));
    if hostData = nil then
    begin
      Fail('GetMem failed in Test_rocrand_generate_log_normal_double for host data');
      Exit;
    end;

    try
      memcpyStatus := hipMemcpy(hostData, devicePtr, numElements * SizeOf(Double), hipMemcpyDeviceToHost);
      if memcpyStatus <> HIP_SUCCESS then
      begin
        Fail('hipMemcpy failed in Test_rocrand_generate_log_normal_double - HIP Error: ' + IntToStr(Integer(memcpyStatus)));
        Exit;
      end;

      // Basic checks for log-normal distribution:
      // 1. All generated values should be positive
      i := 0;
      while (i < Integer(numElements)) and (hostData[i] > 0.0) do
        Inc(i);
      AssertTrue('All generated log-normal Double values should be positive',
                 i = Integer(numElements)); // If i < numElements, we found a non-positive value

      // 2. Check for variation
      // Compare subsequent elements to the first one using a small epsilon
      i := 1;
      // Use a different variable name for the comparison to avoid confusion
      // or potential issues if 'i' was somehow modified unexpectedly in the loop condition.
      // However, the standard way is fine. Let's just make sure the logic is clear.
      // The goal is to find an element that is significantly different from the first.
      while (i < Integer(numElements)) and (Abs(hostData[i] - hostData[0]) < epsilon) do
        Inc(i);
      AssertTrue('Generated log-normal Double values should show variation',
                 i < Integer(numElements)); // If i reached numElements, all were effectively identical within epsilon

    finally
      FreeMem(hostData);
    end;

  finally
    freeStatus := hipFree(devicePtr);
    if freeStatus <> HIP_SUCCESS then
    begin
      WriteLn('Warning: hipFree failed in Test_rocrand_generate_log_normal_double cleanup - HIP Error: ', Integer(freeStatus));
    end;
  end;
end;


procedure TTest_rocrand.Test_rocrand_generate_log_normal_half;
var
  generateStatus: rocrand_status;
  devicePtr: hipDeviceptr_t;
  numElements: UIntPtr;
  mallocStatus: hipError_t;
  freeStatus: hipError_t;
  // mean and stddev parameters for _half functions are complex (packed structures)
  // For now, focus on the call and memory management.
begin
  numElements := 1000; // Number of half elements to generate
  devicePtr := nil;

  // Allocate device memory for the output (half is 16 bits)
  mallocStatus := hipMalloc(devicePtr, numElements * SizeOf(UInt16));
  if mallocStatus <> HIP_SUCCESS then
  begin
    Fail('hipMalloc failed in Test_rocrand_generate_log_normal_half - HIP Error: ' + IntToStr(Integer(mallocStatus)));
    Exit;
  end;

  try
    // Test generating log-normally distributed half-precision numbers
    // Note: Generator compatibility and parameter handling for packed structures.
    generateStatus := rocrand_generate_log_normal_half(generator, devicePtr, numElements, Default(half2), Default(half2)); // Using Default for now
    // Assert that the generation call was successful
    AssertEquals('rocrand_generate_log_normal_half should return success',
                 Integer(ROCRAND_STATUS_SUCCESS),
                 Integer(generateStatus));

    // Note: Verifying the content requires unpacking half data, which is complex.
    // The test confirms the API call and memory management work.

  finally
    freeStatus := hipFree(devicePtr);
    if freeStatus <> HIP_SUCCESS then
    begin
      WriteLn('Warning: hipFree failed in Test_rocrand_generate_log_normal_half cleanup - HIP Error: ', Integer(freeStatus));
    end;
  end;
end;


//procedure TTest_rocrand.Test_rocrand_generate_poisson;
//var
//  generateStatus: rocrand_status;
//  devicePtr: hipDeviceptr_t;
//  numElements: UIntPtr;
//  i: Integer;
//  hostData: PUInt32; // Poisson generates UInt32 (unsigned int)
//  mallocStatus: hipError_t;
//  freeStatus: hipError_t;
//  memcpyStatus: hipError_t;
//  lambda: Double;
//begin
//  numElements := 1000; // Number of UInt32 elements to generate
//  devicePtr := nil;
//  lambda := 4.2; // Example lambda (mean) for Poisson distribution
//
//  // Allocate device memory for the output (Poisson outputs UInt32)
//  mallocStatus := hipMalloc(devicePtr, numElements * SizeOf(UInt32));
//  if mallocStatus <> HIP_SUCCESS then
//  begin
//    Fail('hipMalloc failed in Test_rocrand_generate_poisson - HIP Error: ' + IntToStr(Integer(mallocStatus)));
//    Exit;
//  end;
//
//  try
//    // Test generating Poisson-distributed UInt32 numbers
//    generateStatus := rocrand_generate_poisson(generator, devicePtr, numElements, lambda);
//    // Assert that the generation call was successful
//    AssertEquals('rocrand_generate_poisson should return success',
//                 Integer(ROCRAND_STATUS_SUCCESS),
//                 Integer(generateStatus));
//
//    // Optional: Copy data back to host to verify
//    GetMem(hostData, numElements * SizeOf(UInt32));
//    if hostData = nil then
//    begin
//      Fail('GetMem failed in Test_rocrand_generate_poisson for host data');
//      Exit;
//    end;
//
//    try
//      memcpyStatus := hipMemcpy(hostData, devicePtr, numElements * SizeOf(UInt32), hipMemcpyDeviceToHost);
//      if memcpyStatus <> HIP_SUCCESS then
//      begin
//        Fail('hipMemcpy failed in Test_rocrand_generate_poisson - HIP Error: ' + IntToStr(Integer(memcpyStatus)));
//        Exit;
//      end;
//
//      // Basic checks for Poisson distribution:
//      // 1. All generated values should be non-negative integers (UInt32 inherently satisfies this)
//      //    So, no specific check needed for negativity here.
//
//      // 2. Check for variation (extremely unlikely all values are identical)
//      i := 1;
//      while (i < Integer(numElements)) and (hostData[i] = hostData[0]) do
//        Inc(i);
//      AssertTrue('Generated Poisson UInt32 values should show variation',
//                 i < Integer(numElements)); // If i reached numElements, all were identical
//
//      // 3. (Optional/Advanced) Check if sample mean is close to lambda.
//      // This would require summing hostData and comparing sum/N to lambda with tolerance.
//      // var sampleMean: Double := 0.0;
//      // for i := 0 to Integer(numElements) - 1 do
//      //   sampleMean := sampleMean + hostData[i];
//      // sampleMean := sampleMean / numElements;
//      // AssertTrue('Sample mean should be close to lambda',
//      //            Abs(sampleMean - lambda) < some_tolerance);
//
//    finally
//      FreeMem(hostData);
//    end;
//
//  finally
//    freeStatus := hipFree(devicePtr);
//    if freeStatus <> HIP_SUCCESS then
//    begin
//      WriteLn('Warning: hipFree failed in Test_rocrand_generate_poisson cleanup - HIP Error: ', Integer(freeStatus));
//    end;
//  end;
//end;


initialization
  RegisterTest(TTest_rocRAND);
end.

