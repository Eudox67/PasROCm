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

  Shared helpers for the signature checks: half and bfloat16 conversion,
  CopySign and similar value utilities. Declares no tests of its own.
------------------------------------------------------------------------------}

unit ROCm_test_utils; // Or include in Test_ROCm.pas

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math, rocblas, hip_types, hip_runtime; // Make sure rocblas.pas is accessible

// --- IEEE 754 Compliant FP32 to FP16 Conversion ---
function SingleToHalf(const aValue: Single): rocblas_half;

// --- IEEE 754 Compliant FP16 to FP32 Conversion ---
function HalfToSingle(const aValue: rocblas_half): Single;

// --- IEEE 754 Compliant FP32 to BF16 Conversion ---
function SingleToBFloat16(const aValue: Single): rocblas_bfloat16;

// --- IEEE 754 Compliant BF16 to FP32 Conversion ---
function BFloat16ToSingle(const aValue: rocblas_bfloat16): Single;

{$IF FPC_FULLVERSION < 30302}
// --- IEEE 754 Compliant CopySign ---
function CopySign(const x: Single; const y: Single): Single; overload;
function CopySign(const x: Double; const y: Double): Double; overload;

function IsSameSign(const x: Single; const y: Single): Boolean; overload;
function IsSameSign(const x: Double; const y: Double): Boolean; overload;
{$EndIF}

function CalculateRealNorm(n: Integer; const x: PSingle; incx: Integer): Double; overload;
function CalculateRealNorm(n: Integer; const x: PDouble; incx: Integer): Double; overload;

function CalculateComplexNorm(n: Integer; const x: Procfloat_complex; incx: Integer): Double; overload;
function CalculateComplexNorm(n: Integer; const x: Procdouble_complex; incx: Integer): Double; overload;

procedure ComplexMul(var z: rocblas_float_complex;  const k: rocblas_float_complex);  overload;
procedure ComplexMul(var z: rocblas_double_complex; const k: rocblas_double_complex); overload;

procedure ComplexFMA(var acc: rocblas_float_complex;  const a, x, alpha: rocblas_float_complex);  overload;
procedure ComplexFMA(var acc: rocblas_double_complex; const a, x, alpha: rocblas_double_complex); overload;

procedure ComplexAdd(var z: rocblas_float_complex; const w: rocblas_float_complex); overload;
procedure ComplexAdd(var z: rocblas_double_complex; const w: rocblas_double_complex); overload;

procedure ComplexSub(var z: rocblas_float_complex; const w: rocblas_float_complex); overload;
procedure ComplexSub(var z: rocblas_double_complex; const w: rocblas_double_complex); overload;

procedure ComplexDiv(var z: rocblas_float_complex; const w: rocblas_float_complex); overload;
procedure ComplexDiv(var z: rocblas_double_complex; const w: rocblas_double_complex); overload;

implementation

const
  // FP32 (Single) Bit Layout
  FP32_SIGN_SHIFT     = 31;
  FP32_EXPONENT_SHIFT = 23;
  FP32_MANTISSA_MASK  = $7FFFFF; // 23 bits
  FP32_EXPONENT_BIAS  = 127;

  // FP16 (Half) Bit Layout
  FP16_SIGN_SHIFT     = 15;
  FP16_EXPONENT_SHIFT = 10;
  FP16_MANTISSA_MASK  = $3FF; // 10 bits
  FP16_EXPONENT_BIAS  = 15;

  // BF16 Bit Layout (Shares exponent placement with FP32)
  BF16_SIGN_SHIFT     = 15;
  BF16_EXPONENT_SHIFT = 7;  // Relative to FP16 layout, but absolute position same as FP32
  BF16_MANTISSA_MASK  = $7F; // 7 bits (lower 7 of FP32's 23)
  BF16_EXPONENT_BIAS  = 127; // Same bias as FP32

  // Special FP16 values
  FP16_INFINITY       = $7C00;
  FP16_NEG_INFINITY   = $FC00;
  FP16_QUIET_NAN      = $7E00; // Standard QNaN bit pattern

  // Special BF16 values (Upper 16 bits of corresponding FP32 values)
  BF16_INFINITY       = $7F80;
  BF16_NEG_INFINITY   = $FF80;
  BF16_QUIET_NAN      = $7FC0; // Standard QNaN bit pattern in FP32 truncated


// --- IEEE 754 Compliant FP32 to FP16 Conversion ---
function SingleToHalf(const aValue: Single): rocblas_half;
var
  // Use absolute variable declaration correctly
  f_bits: UInt32;
  f_bits_alias: UInt32 absolute f_bits; // Alias to manipulate bits
  sign: UInt16;
  exponent: Integer; // Needs to handle negative adjusted exponents
  mantissa: UInt32;
  half_sign: UInt16;
  half_exponent: UInt16;
  half_mantissa: UInt16;
  half_bits: UInt16;
  f_exp_raw: UInt32;
  f_mantissa_raw: UInt32;
begin
  // Assign the bit pattern of aValue to f_bits
  f_bits_alias := PUInt32(@aValue)^;

  // Handle special cases first
  if IsNan(aValue) then
  begin
    Result.data := FP16_QUIET_NAN;
    Exit;
  end;
  if IsInfinite(aValue) then
  begin
    if aValue > 0 then
      Result.data := FP16_INFINITY
    else
      Result.data := FP16_NEG_INFINITY;
    Exit;
  end;

  // Extract components from FP32 bit pattern
  sign := (f_bits shr FP32_SIGN_SHIFT) and $1;
  f_exp_raw := (f_bits shr FP32_EXPONENT_SHIFT) and $FF;
  f_mantissa_raw := f_bits and FP32_MANTISSA_MASK;

  // Handle zero (including signed zero)
  if (f_exp_raw = 0) and (f_mantissa_raw = 0) then
  begin
    Result.data := sign shl FP16_SIGN_SHIFT; // +/- zero
    Exit;
  end;

  // Convert exponent from FP32 bias to FP16 bias
  exponent := Integer(f_exp_raw) - FP32_EXPONENT_BIAS + FP16_EXPONENT_BIAS;

  // Prepare mantissa (FP32 has 23 bits, FP16 has 10)
  mantissa := f_mantissa_raw;

  // Handle subnormal FP32 numbers
  if f_exp_raw = 0 then
  begin
    // FP32 subnormal: find the leading '1' bit to normalize
    while (mantissa > 0) and ((mantissa and $00400000) = 0) do // $00400000 = 1 << 22
    begin
      mantissa := mantissa shl 1;
      Dec(exponent); // Adjust exponent as we shift
    end;
    // Shift out the implicit '1' bit that becomes explicit in normal form (if found)
    if mantissa > 0 then
    begin
       mantissa := mantissa shl 1;
       Dec(exponent);
    end;
    // Clear the bit that was the implicit '1' (now shifted out or beyond precision)
    mantissa := mantissa and FP32_MANTISSA_MASK;
  end;

  // Round mantissa to fit into 10 bits
  // We are rounding the 23-bit mantissa to 10 bits.
  // Bit 12 (0-based from LSB) is the 0.5 bit for rounding.
  // Bit 13 is the 1.0 bit.
  if (mantissa and $1FFF) > $1000 then // If bits 12..0 are > 0.5 (i.e., bit 12 is 1 and lower bits are non-zero, or higher bits set)
    mantissa := mantissa + $2000 // Add 1 to bit 13 (0.5 -> 1.0) for rounding up
  else if (mantissa and $3FFF) = $1000 then // If bits 13..0 are exactly 0.5 (bit 12 is 1, bits 11..0 are 0)
    if (mantissa and $2000) <> 0 then // If bit 13 (0.5 place) is 1 (odd), round up to even
      mantissa := mantissa + $2000;

  // Shift mantissa down to 10 bits (right by 13)
  mantissa := mantissa shr 13;

  // Handle overflow (exponent too large for FP16)
  if exponent >= 31 then
  begin
    // Overflow to infinity
    Result.data := (sign shl FP16_SIGN_SHIFT) or FP16_INFINITY;
    Exit;
  end;

  // Handle underflow (exponent too small for FP16 normal)
  if exponent <= 0 then
  begin
    // Underflow to subnormal or zero
    if exponent < -10 then
    begin
      // Too small, rounds to zero
      Result.data := sign shl FP16_SIGN_SHIFT;
      Exit;
    end;

    // Create FP16 subnormal
    // Mantissa needs to be shifted right based on how much underflow
    // The implicit '1' is not used for subnormals.
    // Adjust mantissa for subnormal representation.
    // Add the implicit 1 back temporarily for correct shifting
    mantissa := (1 shl 10) or (mantissa and FP16_MANTISSA_MASK);
    // Shift right by (1 - exponent) positions to represent the correct value
    // e.g., if exponent=-1, shift right by 2.
    mantissa := mantissa shr (1 - exponent);

    // Finalize subnormal parts
    half_mantissa := mantissa and FP16_MANTISSA_MASK;
    half_exponent := 0; // Exponent is 0 for subnormals
  end
  else
  begin
    // Normal FP16 number
    half_exponent := exponent and $1F; // 5 bits
    half_mantissa := mantissa and FP16_MANTISSA_MASK; // 10 bits
  end;

  // Assemble final FP16 bits
  half_sign := sign and $1;
  half_bits := (half_sign shl FP16_SIGN_SHIFT) or (half_exponent shl FP16_EXPONENT_SHIFT) or half_mantissa;
  Result.data := half_bits;
end;


// --- IEEE 754 Compliant FP16 to FP32 Conversion ---
function HalfToSingle(const aValue: rocblas_half): Single;
var
  h_bits: UInt16;
  sign: UInt32;
  // Use PUInt32 absolute for the result to reinterpret bits
  f_bits: UInt32;
  f_bits_alias: UInt32 absolute f_bits; // Alias for bit manipulation
  f_sign: UInt32;
  f_exponent: UInt32;
  f_mantissa: UInt32;
  h_exp_raw: UInt16;
  h_mantissa_raw: UInt16;
begin
  h_bits := aValue.data;

  // Handle special cases first
  if (h_bits and $7FFF) > $7C00 then
  begin
    // NaN: Set FP32 QNaN bit for consistency if needed
    f_bits := (h_bits and $8000) shl 16; // Sign
    f_bits := f_bits or $7FC00000;       // QNaN exponent and standard QNaN bit
    f_bits := f_bits or ((h_bits and $3FF) shl 13); // Propagate lower bits of mantissa
    // Reinterpret f_bits as Single using absolute variable
    Result := PSingle(@f_bits_alias)^;
    Exit;
  end;

  if (h_bits and $7FFF) = $7C00 then
  begin
    // Infinity
    f_bits := (h_bits and $8000) shl 16; // Sign
    f_bits := f_bits or $7F800000;       // Infinity exponent
    // Reinterpret f_bits as Single
    Result := PSingle(@f_bits_alias)^;
    Exit;
  end;

  // Extract components from FP16
  sign := (h_bits shr FP16_SIGN_SHIFT) and $1;
  h_exp_raw := (h_bits shr FP16_EXPONENT_SHIFT) and $1F;
  h_mantissa_raw := h_bits and FP16_MANTISSA_MASK;

  // Handle zero (including signed zero)
  if (h_exp_raw = 0) and (h_mantissa_raw = 0) then
  begin
    f_bits := sign shl 31; // Set sign bit in FP32 position, rest zero
    // Reinterpret f_bits as Single
    Result := PSingle(@f_bits_alias)^;
    Exit;
  end;

  // Convert exponent from FP16 bias to FP32 bias
  // Handle subnormal FP16 separately
  if h_exp_raw = 0 then
  begin
    // FP16 subnormal
    f_exponent := 0; // Will be adjusted after normalization
    f_mantissa := h_mantissa_raw; // 10-bit mantissa
    // Normalize the subnormal FP16 mantissa for FP32
    // Find the leading 1 bit in the 10-bit mantissa
    while (f_mantissa > 0) and ((f_mantissa and $00000200) = 0) do // $00000200 = 1 << 9 (highest bit of 10-bit mantissa)
    begin
      f_mantissa := f_mantissa shl 1;
      Dec(f_exponent); // Decrement exponent as we shift left
    end;
    // Shift left one more time to remove the implicit leading 1 (which becomes explicit in normalized FP32)
    if f_mantissa > 0 then
    begin
       f_mantissa := f_mantissa shl 1;
       Dec(f_exponent);
    end;
    // Clear the bit that was the implicit '1'
    f_mantissa := f_mantissa and $000003FF; // Keep only 10 bits relevant now
    // Adjust exponent for FP32 bias
    f_exponent := Integer(f_exponent) - FP16_EXPONENT_BIAS + FP32_EXPONENT_BIAS;
  end
  else
  begin
    // Normal FP16 number
    f_exponent := (Integer(h_exp_raw) - FP16_EXPONENT_BIAS) + FP32_EXPONENT_BIAS;
    f_mantissa := h_mantissa_raw; // 10-bit mantissa
  end;

  // Shift FP16 mantissa (10 bits) up to FP32 mantissa size (23 bits)
  f_mantissa := f_mantissa shl 13; // Shift left by 13

  // Clamp final exponent if necessary (should be handled by prior logic for valid FP16)
  if f_exponent >= 255 then
    f_exponent := 255 // FP32 Infinity
  else if Integer(f_exponent) <= 0 then
    f_exponent := 0; // FP32 Zero/Subnormal (already handled, but safe)

  // Assemble final FP32 bits
  f_sign := sign and $1;
  f_bits := (f_sign shl FP32_SIGN_SHIFT) or (f_exponent shl FP32_EXPONENT_SHIFT) or (f_mantissa and FP32_MANTISSA_MASK);

  // Reinterpret f_bits as Single using absolute variable
  Result := PSingle(@f_bits_alias)^;
end;


// --- IEEE 754 Compliant FP32 to BF16 Conversion ---
function SingleToBFloat16(const aValue: Single): rocblas_bfloat16;
var
  // Use absolute variable declaration correctly
  f_bits: UInt32;
  f_bits_alias: UInt32 absolute f_bits; // Alias to manipulate bits
  rounded_bits: UInt32;
  bf16_bits: UInt16;
begin
  // Assign the bit pattern of aValue to f_bits
  f_bits_alias := PUInt32(@aValue)^;

  // Handle special cases first
  if IsNan(aValue) then
  begin
    // Simple approach: truncate and ensure it's recognized as NaN.
    // More sophisticated handling could preserve QNaN/SNaN if needed.
    bf16_bits := UInt16((f_bits shr 16) and $FFFF);
    // Ensure it's a NaN in BF16 space if not already obvious
    if (bf16_bits and $7FFF) < $7F80 then // Not infinity or NaN yet
      bf16_bits := bf16_bits or $40; // Set a standard QNaN bit (bit 6 in mantissa)
    Result.data := bf16_bits;
    Exit;
  end;
  if IsInfinite(aValue) then
  begin
    if aValue > 0 then
      Result.data := BF16_INFINITY
    else
      Result.data := BF16_NEG_INFINITY;
    Exit;
  end;

  // Perform IEEE 754 round-to-nearest-even:
  // Add 0x8000 (0.5 in the truncated 16-bit space) to the original 32-bit representation.
  rounded_bits := f_bits + $8000;

  // Handle potential overflow from rounding (carry into exponent/sign)
  if (rounded_bits and $7F800000) = $7F800000 then
  begin
    // Rounding caused overflow to infinity
    if aValue > 0 then
      Result.data := BF16_INFINITY
    else
      Result.data := BF16_NEG_INFINITY;
    Exit;
  end;

  // Truncate to upper 16 bits (BF16 representation)
  bf16_bits := UInt16(rounded_bits shr 16);
  Result.data := bf16_bits;
end;


// --- IEEE 754 Compliant BF16 to FP32 Conversion ---
function BFloat16ToSingle(const aValue: rocblas_bfloat16): Single;
var
  f_bits: UInt32;
  f_bits_alias: UInt32 absolute f_bits; // Alias for reinterpretation
begin
  // Place the 16 bits in the upper part and zero the lower 16 bits
  f_bits := UInt32(aValue.data) shl 16;
  // Reinterpret f_bits as Single using absolute variable
  Result := PSingle(@f_bits_alias)^;
end;

{$IF FPC_FULLVERSION < 30302}
function CopySign(const x: Single; const y: Single): Single;
type
  // Use a type alias for reinterpretation
  PUInt32 = ^UInt32;
var
  bits_x, bits_y: UInt32;
  result_bits: UInt32;
begin
  // Reinterpret the bits of the floating point numbers
  bits_x := PUInt32(@x)^;
  bits_y := PUInt32(@y)^;

  // Clear the sign bit of x, and take only the sign bit of y
  result_bits := (bits_x and $7FFFFFFF) or (bits_y and $80000000);

  // Reinterpret the resulting bits back to a Single
  Result := PSingle(@result_bits)^;
end;

function CopySign(const x: Double; const y: Double): Double;
type
  // Use a type alias for reinterpretation
  PUInt64 = ^UInt64;
var
  bits_x, bits_y: UInt64;
  result_bits: UInt64;
begin
  // Reinterpret the bits of the floating point numbers
  bits_x := PUInt64(@x)^;
  bits_y := PUInt64(@y)^;

  // Clear the sign bit of x, and take only the sign bit of y
  result_bits := (bits_x and $7FFFFFFFFFFFFFFF) or (bits_y and $8000000000000000);

  // Reinterpret the resulting bits back to a Double
  Result := PDouble(@result_bits)^;
end;

function IsSameSign(const x: Single; const y: Single): Boolean;
type
  PUInt32 = ^UInt32;
var
  bits_x, bits_y: UInt32;
begin
  // Reinterpret the bits of the floating point numbers
  bits_x := PUInt32(@x)^;
  bits_y := PUInt32(@y)^;

  // The signs are the same if the XOR of the most significant bits is 0.
  // This is equivalent to checking if the XOR of the full numbers is non-negative.
  Result := ((bits_x xor bits_y) and $80000000) = 0;
end;

function IsSameSign(const x: Double; const y: Double): Boolean;
type
  PUInt64 = ^UInt64;
var
  bits_x, bits_y: UInt64;
begin
  // Reinterpret the bits of the floating point numbers
  bits_x := PUInt64(@x)^;
  bits_y := PUInt64(@y)^;

  // The signs are the same if the XOR of the most significant bits is 0.
  Result := ((bits_x xor bits_y) and $8000000000000000) = 0;
end;
{$ENDIF}

// Helper function to calculate the Euclidean norm (l2 norm) of a real vector
function CalculateRealNorm(n: Integer; const x: PSingle; incx: Integer): Double; overload;
var
  i, idx: Integer;
  sum: Double;
  temp: Double;
begin
  sum := 0.0;
  idx := 0;
  for i := 0 to n - 1 do
  begin
    temp := x[idx];
    sum := sum + (temp * temp);
    idx := idx + incx;
  end;
  Result := Sqrt(sum);
end;

function CalculateRealNorm(n: Integer; const x: PDouble; incx: Integer): Double; overload;
var
  i, idx: Integer;
  sum: Double;
  temp: Double;
begin
  sum := 0.0;
  idx := 0;
  for i := 0 to n - 1 do
  begin
    temp := x[idx];
    sum := sum + (temp * temp);
    idx := idx + incx;
  end;
  Result := Sqrt(sum);
end;

// Helper function to calculate the Euclidean norm (l2 norm) of a complex vector
function CalculateComplexNorm(n: Integer; const x: Procfloat_complex; incx: Integer): Double; overload;
var
  i, idx: Integer;
  sum: Double;
  temp_re, temp_im: Double;
begin
  sum := 0.0;
  idx := 0;
  for i := 0 to n - 1 do
  begin
    temp_re := x[idx].x;
    temp_im := x[idx].y;
    sum := sum + (temp_re * temp_re) + (temp_im * temp_im);
    idx := idx + incx;
  end;
  Result := Sqrt(sum);
end;

function CalculateComplexNorm(n: Integer; const x: Procdouble_complex; incx: Integer): Double; overload;
var
  i, idx: Integer;
  sum: Double;
  temp_re, temp_im: Double;
begin
  sum := 0.0;
  idx := 0;
  for i := 0 to n - 1 do
  begin
    temp_re := x[idx].x;
    temp_im := x[idx].y;
    sum := sum + (temp_re * temp_re) + (temp_im * temp_im);
    idx := idx + incx;
  end;
  Result := Sqrt(sum);
end;

procedure ComplexMul(var z: rocblas_float_complex;  const k: rocblas_float_complex);  overload; inline;
var
  r, i: Single;
begin
  r := z.x * k.x - z.y * k.y;
  i := z.x * k.y + z.y * k.x;
  z.x := r; z.y := i;
end;

procedure ComplexMul(var z: rocblas_double_complex; const k: rocblas_double_complex); overload; inline;
var
  r, i: Double;
begin
  r := z.x * k.x - z.y * k.y;
  i := z.x * k.y + z.y * k.x;
  z.x := r; z.y := i;
end;

procedure ComplexFMA(var acc: rocblas_float_complex;  const a, x, alpha: rocblas_float_complex);  overload; inline;
var
  p: rocblas_float_complex;
begin
  // p := a * x
  p.x := a.x * x.x - a.y * x.y;
  p.y := a.x * x.y + a.y * x.x;

  // p := p * alpha
  ComplexMul(p, alpha);

  // acc := acc + p
  acc.x := acc.x + p.x;
  acc.y := acc.y + p.y;
end;

procedure ComplexFMA(var acc: rocblas_double_complex; const a, x, alpha: rocblas_double_complex); overload; inline;
var
  p: rocblas_double_complex;
begin
  p.x := a.x * x.x - a.y * x.y;
  p.y := a.x * x.y + a.y * x.x;
  ComplexMul(p, alpha);
  acc.x := acc.x + p.x;
  acc.y := acc.y + p.y;
end;

procedure ComplexAdd(var z: rocblas_float_complex; const w: rocblas_float_complex); overload;
begin
  z.x := z.x + w.x;
  z.y := z.y + w.y;
end;

procedure ComplexAdd(var z: rocblas_double_complex; const w: rocblas_double_complex); overload;
begin
  z.x := z.x + w.x;
  z.y := z.y + w.y;
end;

procedure ComplexSub(var z: rocblas_float_complex; const w: rocblas_float_complex); overload;
begin
  z.x := z.x - w.x;
  z.y := z.y - w.y;
end;

procedure ComplexSub(var z: rocblas_double_complex; const w: rocblas_double_complex); overload;
begin
  z.x := z.x - w.x;
  z.y := z.y - w.y;
end;

procedure ComplexDiv(var z: rocblas_float_complex; const w: rocblas_float_complex); overload;
var
  d, r, i: Single;
begin
  d := w.x * w.x + w.y * w.y;
  if d = 0.0 then Exit; // Avoid division by zero
  r := (z.x * w.x + z.y * w.y) / d;
  i := (z.y * w.x - z.x * w.y) / d;
  z.x := r;
  z.y := i;
end;

procedure ComplexDiv(var z: rocblas_double_complex; const w: rocblas_double_complex); overload;
var
  d, r, i: Double;
begin
  d := w.x * w.x + w.y * w.y;
  if d = 0.0 then Exit; // Avoid division by zero
  r := (z.x * w.x + z.y * w.y) / d;
  i := (z.y * w.x - z.x * w.y) / d;
  z.x := r;
  z.y := i;
end;


end.
