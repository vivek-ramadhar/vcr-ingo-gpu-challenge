#include <cstdint>
#include <cuda.h>
#include <stdexcept>

#include "/home/vivek/CGBN/include/cgbn/cgbn.cu"
#include "/home/vivek/CGBN/include/cgbn/cgbn_cuda.h"

namespace ptx {

// **************************
//  ADD/MUL FOR 32-bit LIMBS
// **************************

__device__ __forceinline__ uint32_t add_cc(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t addc(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm volatile("addc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t addc_cc(const uint32_t x,
                                            const uint32_t y) {
  uint32_t result;
  asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t sub_cc(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t subc(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm volatile("subc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t subc_cc(const uint32_t x,
                                            const uint32_t y) {
  uint32_t result;
  asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t mul_lo(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm("mul.lo.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t mul_hi(const uint32_t x, const uint32_t y) {
  uint32_t result;
  asm("mul.hi.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
  return result;
}

__device__ __forceinline__ uint32_t mad_lo_cc(const uint32_t x,
                                              const uint32_t y,
                                              const uint32_t z) {
  uint32_t result;
  asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;"
               : "=r"(result)
               : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ __forceinline__ uint32_t madc_hi(const uint32_t x, const uint32_t y,
                                            const uint32_t z) {
  uint32_t result;
  asm volatile("madc.hi.u32 %0, %1, %2, %3;"
               : "=r"(result)
               : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ __forceinline__ uint32_t madc_lo_cc(const uint32_t x,
                                               const uint32_t y,
                                               const uint32_t z) {
  uint32_t result;
  asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;"
               : "=r"(result)
               : "r"(x), "r"(y), "r"(z));
  return result;
}

__device__ __forceinline__ uint32_t madc_hi_cc(const uint32_t x,
                                               const uint32_t y,
                                               const uint32_t z) {
  uint32_t result;
  asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;"
               : "=r"(result)
               : "r"(x), "r"(y), "r"(z));
  return result;
}

// **************************
//  ADD/MUL FOR 64-bit LIMBS
// **************************

__device__ __forceinline__ uint64_t mad_lo_cc(const uint64_t x,
                                              const uint64_t y,
                                              const uint64_t z) {
  uint64_t result;
  asm volatile("mad.lo.cc.u64 %0, %1, %2, %3;"
               : "=l"(result)
               : "l"(x), "l"(y), "l"(z));
  return result;
}
__device__ __forceinline__ uint64_t madc_hi(const uint64_t x, const uint64_t y,
                                            const uint64_t z) {
  uint64_t result;
  asm volatile("madc.hi.u64 %0, %1, %2, %3;"
               : "=l"(result)
               : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ __forceinline__ uint64_t madc_lo_cc(const uint64_t x,
                                               const uint64_t y,
                                               const uint64_t z) {
  uint64_t result;
  asm volatile("madc.lo.cc.u64 %0, %1, %2, %3;"
               : "=l"(result)
               : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ __forceinline__ uint64_t madc_hi_cc(const uint64_t x,
                                               const uint64_t y,
                                               const uint64_t z) {
  uint64_t result;
  asm volatile("madc.hi.cc.u64 %0, %1, %2, %3;"
               : "=l"(result)
               : "l"(x), "l"(y), "l"(z));
  return result;
}

__device__ __forceinline__ uint64_t mul_lo(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm("mul.lo.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t mul_hi(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm("mul.hi.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t add_cc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("add.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t addc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("addc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

__device__ __forceinline__ uint64_t addc_cc(const uint64_t x,
                                            const uint64_t y) {
  uint64_t result;
  asm volatile("addc.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

} // namespace ptx

template <typename WordT, int N> struct __align__(16) bigint_t {
  WordT limbs[N];
};

template <typename WordT> constexpr int tlc() {
  return 256 / (sizeof(WordT) * 8);
}

using bigint = bigint_t<uint32_t, 8>;       // 256-bit, 32-bit limbs
using bigint_wide = bigint_t<uint32_t, 16>; // 512-bit, 32-bit limbs

using bigint64 = bigint_t<uint64_t, 4>;      // 256-bit, 64-bit limbs
using bigint64_wide = bigint_t<uint64_t, 8>; // 512-bit, 64-bit limbs

//"total limbs count"
template <typename WordT, int TLC>
static __device__ __forceinline__ void mul_n(WordT *acc, const WordT *a,
                                             WordT bi) {
#pragma unroll
  for (size_t i = 0; i < TLC; i += 2) {
    acc[i] = ptx::mul_lo(a[i], bi);
    acc[i + 1] = ptx::mul_hi(a[i], bi);
  }
}

template <typename WordT, int TLC>
static __device__ __forceinline__ void cmad_n(WordT *acc, const WordT *a,
                                              WordT bi, size_t n = TLC) {
  acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
  acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);

#pragma unroll
  for (size_t i = 2; i < n; i += 2) {
    acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
    acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
  }
}

template <typename WordT, int TLC>
static __device__ __forceinline__ void mad_row(WordT *odd, WordT *even,
                                               const WordT *a, WordT bi) {
  cmad_n<WordT, TLC>(odd, a + 1, bi, TLC - 2);
  odd[TLC - 2] = ptx::madc_lo_cc(a[TLC - 1], bi, 0);
  odd[TLC - 1] = ptx::madc_hi(a[TLC - 1], bi, 0);
  cmad_n<WordT, TLC>(even, a, bi, TLC);
  odd[TLC - 1] = ptx::addc(odd[TLC - 1], 0);
}

template <typename WordT, int TLC>
static __device__ __forceinline__ void
multiply_raw_device(const bigint_t<WordT, TLC> &as,
                    const bigint_t<WordT, TLC> &bs,
                    bigint_t<WordT, 2 * TLC> &rs) {
  const WordT *a = as.limbs;
  const WordT *b = bs.limbs;
  WordT *even = rs.limbs;
  __align__(8) WordT odd[2 * TLC - 2];
  mul_n<WordT, TLC>(even, a, b[0]);
  mul_n<WordT, TLC>(odd, a + 1, b[0]);
  mad_row<WordT, TLC>(&even[2], &odd[0], a, b[1]);
  size_t i;
#pragma unroll
  for (i = 2; i < TLC - 1; i += 2) {
    mad_row<WordT, TLC>(&odd[i], &even[i], a, b[i]);
    mad_row<WordT, TLC>(&even[i + 2], &odd[i], a, b[i + 1]);
  }
  // merge |even| and |odd|
  even[1] = ptx::add_cc(even[1], odd[0]);
  for (i = 1; i < 2 * TLC - 2; i++)
    even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
  even[i + 1] = ptx::addc(even[i + 1], 0);
}

static __device__ __forceinline__ void addcc_128(bigint_t<uint32_t, 4> &a,
                                                 bigint_t<uint32_t, 4> &b,
                                                 bigint_t<uint32_t, 4> &r,
                                                 uint32_t &carry) {
  uint32_t r0, r1, r2, r3, c;

  asm volatile("add.cc.u32	%0, %5, %9;  \n\t"
               "addc.cc.u32	%1, %6, %10; \n\t"
               "addc.cc.u32	%2, %7, %11; \n\t"
               "addc.cc.u32	%3, %8, %12; \n\t"
               "addc.u32	%4, 0, 0;	"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(c)
               : "r"(a.limbs[0]), "r"(a.limbs[1]), "r"(a.limbs[2]),
                 "r"(a.limbs[3]), "r"(b.limbs[0]), "r"(b.limbs[1]),
                 "r"(b.limbs[2]), "r"(b.limbs[3]));

  r.limbs[0] = r0;
  r.limbs[1] = r1;
  r.limbs[2] = r2;
  r.limbs[3] = r3;
  carry = c;
}

static __device__ __forceinline__ void subcc_256(bigint_t<uint32_t, 8> &a,
                                                 bigint_t<uint32_t, 8> &b,
                                                 bigint_t<uint32_t, 8> &r,
                                                 uint32_t &carry) {
  bigint_t<uint32_t, 8> rtmp = {0};
  rtmp.limbs[0] = ptx::sub_cc(a.limbs[0], b.limbs[0]);
#pragma unroll
  for (int i = 1; i < 8; i++) {
    rtmp.limbs[i] = ptx::subc_cc(a.limbs[i], b.limbs[i]);
  }
  carry = ptx::subc(carry, 0);
  r = rtmp;
}

// x = x_lo + cx*2^128
// y = y_lo + cy*2^128
// cx/cy = 0/1
static __device__ __forceinline__ void
mulc_cc_128(bigint_t<uint32_t, 4> &x, uint32_t &cx, bigint_t<uint32_t, 4> &y,
            uint32_t &cy, bigint_t<uint32_t, 8> &result, uint32_t &carry) {
  bigint_t<uint32_t, 8> tmp = {0};
  uint32_t tmpc = 0;
  // (x_lo*y_lo)
  multiply_raw_device<uint32_t, 4>(x, y, tmp);

  // cy * 2^128 * x_lo
  tmp.limbs[4] = ptx::add_cc(tmp.limbs[4], ptx::mul_lo(cy, x.limbs[0]));
  tmp.limbs[5] = ptx::addc_cc(tmp.limbs[5], ptx::mul_lo(cy, x.limbs[1]));
  tmp.limbs[6] = ptx::addc_cc(tmp.limbs[6], ptx::mul_lo(cy, x.limbs[2]));
  tmp.limbs[7] = ptx::addc_cc(tmp.limbs[7], ptx::mul_lo(cy, x.limbs[3]));
  tmpc = ptx::addc(tmpc, 0);

  // cx * 2^128 * y_lo
  tmp.limbs[4] = ptx::add_cc(tmp.limbs[4], ptx::mul_lo(cx, y.limbs[0]));
  tmp.limbs[5] = ptx::addc_cc(tmp.limbs[5], ptx::mul_lo(cx, y.limbs[1]));
  tmp.limbs[6] = ptx::addc_cc(tmp.limbs[6], ptx::mul_lo(cx, y.limbs[2]));
  tmp.limbs[7] = ptx::addc_cc(tmp.limbs[7], ptx::mul_lo(cx, y.limbs[3]));
  tmpc = ptx::addc(tmpc, 0);

  // cx * cy * 2^256
  tmpc += ptx::mul_lo(cx, cy);

  result = tmp;
  carry = tmpc;
}

static __device__ __forceinline__ void
karatsuba_raw_device(const bigint &x, const bigint &y, bigint_wide &r) {
  // x = x1*2^128 + x0
  // y = y1*2^128 + y0
  bigint_t<uint32_t, 4> x_lo, x_hi, y_lo, y_hi;
  for (int i = 0; i < 4; i++) {
    x_lo.limbs[i] = x.limbs[i];
    x_hi.limbs[i] = x.limbs[i + 4];
    y_lo.limbs[i] = y.limbs[i];
    y_hi.limbs[i] = y.limbs[i + 4];
  }

  // z3 = (x0 + x1)(y0 + y1)
  // x0 + x1 / y0 + y1 < 2^129 - 1 -> 129-bit result -> 128-bit with carry limb
  uint32_t cx, cy;
  bigint_t<uint32_t, 4> x01, y01;
  addcc_128(x_lo, x_hi, x01, cx);
  addcc_128(y_lo, y_hi, y01, cy);

  bigint_t<uint32_t, 8> z3;
  uint32_t z3c;
  mulc_cc_128(x01, cx, y01, cy, z3, z3c);

  bigint_t<uint32_t, 8> z0, z2;

  // z0 = x0y0
  multiply_raw_device<uint32_t, 4>(x_lo, y_lo, z0);

  bigint_t<uint32_t, 16> xy;
  xy.limbs[0] = z0.limbs[0];
  xy.limbs[1] = z0.limbs[1];
  xy.limbs[2] = z0.limbs[2];
  xy.limbs[3] = z0.limbs[3];

  // z2 = x1y1
  multiply_raw_device<uint32_t, 4>(x_hi, y_hi, z2);

  // z1 = z3 - z2 - z0
  subcc_256(z3, z2, z3, z3c); // prop carry limb through subtractions
  subcc_256(z3, z0, z3, z3c);

  // z2 * 2^256 + z1*2^128 + z0
  // eval with carry limb
  xy.limbs[4] = ptx::add_cc(z0.limbs[4], z3.limbs[0]);
  xy.limbs[5] = ptx::addc_cc(z0.limbs[5], z3.limbs[1]);
  xy.limbs[6] = ptx::addc_cc(z0.limbs[6], z3.limbs[2]);
  xy.limbs[7] = ptx::addc_cc(z0.limbs[7], z3.limbs[3]);

  xy.limbs[8] = ptx::addc_cc(z3.limbs[4], z2.limbs[0]);
  xy.limbs[9] = ptx::addc_cc(z3.limbs[5], z2.limbs[1]);
  xy.limbs[10] = ptx::addc_cc(z3.limbs[6], z2.limbs[2]);
  xy.limbs[11] = ptx::addc_cc(z3.limbs[7], z2.limbs[3]);

  xy.limbs[12] = ptx::addc_cc(z3c, z2.limbs[4]);
  xy.limbs[13] = ptx::addc_cc(0, z2.limbs[5]);
  xy.limbs[14] = ptx::addc_cc(0, z2.limbs[6]);
  xy.limbs[15] = ptx::addc(0, z2.limbs[7]);

  r = xy;
}

const int TLC_32 = 8;

static __device__ __forceinline__ void
add_limbs_device(const uint32_t *x, const uint32_t *y, uint32_t *r) {
  r[0] = ptx::add_cc(x[0], y[0]);
  for (unsigned i = 1; i < (TLC_32 - 1); i++)
    r[i] = ptx::addc_cc(x[i], y[i]);
  r[TLC_32 - 1] = ptx::addc(x[TLC_32 - 1], y[TLC_32 - 1]);
}

template <typename WordT, int TLC>
static __device__ __forceinline__ void
add_limbs_device(const WordT *x, const WordT *y, WordT *r) {
  r[0] = ptx::add_cc(x[0], y[0]);
  for (unsigned i = 1; i < (TLC - 1); i++) {
    r[i] = ptx::addc_cc(x[i], y[i]);
  }
  r[TLC - 1] = ptx::addc(x[TLC - 1], y[TLC - 1]);
}

// a method to create a 256-bit number from 512-bit result to be able to
// perpetually repeat the multiplication using registers
bigint __device__ __forceinline__ get_256_bit_result(const bigint_wide &xs) {
  const uint32_t *x = xs.limbs;
  bigint out{};
  add_limbs_device(x, &x[TLC_32], out.limbs);
  return out;
}

template <typename WordT, int TLC>
bigint_t<WordT, TLC> __device__ __forceinline__
get_256_bit_result(const bigint_t<WordT, 2 * TLC> &xs) {
  const WordT *x = xs.limbs;
  bigint_t<WordT, TLC> out{};
  add_limbs_device<WordT, TLC>(x, &x[TLC], out.limbs);
  return out;
}

// The kernel that does element-wise multiplication of arrays in1 and in2 N
// times
template <int N>
__global__ void multKaratsubaKernel(bigint *in1, const bigint *in2,
                                    bigint_wide *out, size_t n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) {
    bigint i1 = in1[tid];
    const bigint i2 = in2[tid];
    bigint_wide o = {0};
    for (int i = 0; i < N - 1; i++) {
      karatsuba_raw_device(i1, i2, o);
      i1 = get_256_bit_result(o);
    }
    karatsuba_raw_device(i1, i2, out[tid]);
  }
}

template <typename WordT, int TLC, int N>
__global__ void multVectorsKernel(bigint_t<WordT, TLC> *in1,
                                  const bigint_t<WordT, TLC> *in2,
                                  bigint_t<WordT, 2 * TLC> *out, size_t n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) {
    bigint_t<WordT, TLC> i1 = in1[tid];
    const bigint_t<WordT, TLC> i2 = in2[tid];
    bigint_t<WordT, 2 * TLC> o = {0};
    for (int i = 0; i < N - 1; i++) {
      multiply_raw_device<WordT, TLC>(i1, i2, o);
      i1 = get_256_bit_result<WordT, TLC>(o);
    }

    bigint_wide tmp = {0};
    multiply_raw_device<WordT, TLC>(i1, i2, tmp);
    out[tid] = tmp;
  }
}

template <int N>
int mult_vectors_karat(bigint in1[], const bigint in2[], bigint_wide *out,
                       size_t n) {
  // Set the grid and block dimensions
  int threads_per_block = 128;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block + 1;

  multKaratsubaKernel<N><<<num_blocks, threads_per_block>>>(in1, in2, out, n);

  return 0;
}

template <typename WordT, int N>
int mult_vectors(bigint_t<WordT, tlc<WordT>()> in1[],
                 const bigint_t<WordT, tlc<WordT>()> in2[],
                 bigint_t<WordT, 2 * tlc<WordT>()> *out, size_t n) {

  int threads_per_block = 128;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block + 1;

  multVectorsKernel<WordT, tlc<WordT>(), N>
      <<<num_blocks, threads_per_block>>>(in1, in2, out, n);

  return 0;
}

extern "C" int multiply_test(bigint in1[], const bigint in2[], bigint_wide *out,
                             size_t n) {
  try {
    // mult_vectors<uint64_t, 1>(reinterpret_cast<bigint64 *>(in1),
    //                           reinterpret_cast<const bigint64 *>(in2),
    //                           reinterpret_cast<bigint64_wide *>(out), n);
    mult_vectors_karat<1>(in1, in2, out, n);
    return CUDA_SUCCESS;
  } catch (const std::runtime_error &ex) {
    return -1;
  }
}

extern "C" int multiply_bench(bigint in1[], const bigint in2[],
                              bigint_wide *out, size_t n) {
  try {
    // for benchmarking, we need to give each thread a number of multiplication
    // tasks that would ensure that we're mostly measuring compute and not
    // global memory accesses, which is why we do 500 multiplications here
    // mult_vectors<uint64_t, 500>(reinterpret_cast<bigint64 *>(in1),
    //                             reinterpret_cast<const bigint64 *>(in2),
    //                             reinterpret_cast<bigint64_wide *>(out), n);
    mult_vectors_karat<500>(in1, in2, out, n);
    return CUDA_SUCCESS;
  } catch (const std::runtime_error &ex) {
    return -1;
  }
}
