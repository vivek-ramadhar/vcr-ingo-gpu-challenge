#include <cstdint>
#include <cuda.h>
#include <stdexcept>


struct __align__(16) bigint {
    uint32_t limbs[8];
};

struct __align__(16) bigint_wide {
    uint32_t limbs[16];
};

const int TLC = 8;

// Single consolidated PTX block — direct translation of the original
// mul_n / cmad_n / mad_row / merge algorithm.
//
// The original scattered volatile asm approach:
//   - Each ptx:: helper is a separate asm volatile block
//   - volatile forces ordering but creates an optimization barrier at every call
//   - PTXAS cannot see CC register dependencies across block boundaries
//   - Compiler treats every volatile block as an opaque side-effecting black box
//
// This version:
//   - One block: PTXAS sees the full dependency graph
//   - No volatile needed — CC dependencies are internal and handled by PTXAS
//   - Scheduler can interleave independent instructions across what were barriers
//   - Identical algorithm and identical output, just removing the fences
//
// Operand map:
//   %0..%15  : outputs  → rs.limbs[0..15]
//   %16..%23 : inputs   → as.limbs[0..7]  (a0..a7)
//   %24..%31 : inputs   → bs.limbs[0..7]  (b0..b7)

static __device__ __forceinline__ void multiply_raw_device(const bigint &as, const bigint &bs, bigint_wide &rs)
{
    asm(
        "{\n\t"

        // ── Internal register declarations ──────────────────────────────────
        // Named registers are scoped to this block. The mov.b32 load/store
        // instructions at the top and bottom are free — PTXAS eliminates them
        // as pure register aliases.
        ".reg .u32 a0, a1, a2, a3, a4, a5, a6, a7;\n\t"
        ".reg .u32 b0, b1, b2, b3, b4, b5, b6, b7;\n\t"
        ".reg .u32 r0,  r1,  r2,  r3,  r4,  r5,  r6,  r7;\n\t"
        ".reg .u32 r8,  r9,  r10, r11, r12, r13, r14, r15;\n\t"
        // odd accumulator array: 2*TLC - 2 = 14 limbs
        ".reg .u32 o0,  o1,  o2,  o3,  o4,  o5,  o6,  o7;\n\t"
        ".reg .u32 o8,  o9,  o10, o11, o12, o13;\n\t"

        // ── Load inputs ─────────────────────────────────────────────────────
        "mov.b32  a0, %16;  mov.b32  a1, %17;  mov.b32  a2, %18;  mov.b32  a3, %19;\n\t"
        "mov.b32  a4, %20;  mov.b32  a5, %21;  mov.b32  a6, %22;  mov.b32  a7, %23;\n\t"
        "mov.b32  b0, %24;  mov.b32  b1, %25;  mov.b32  b2, %26;  mov.b32  b3, %27;\n\t"
        "mov.b32  b4, %28;  mov.b32  b5, %29;  mov.b32  b6, %30;  mov.b32  b7, %31;\n\t"

        // ════════════════════════════════════════════════════════════════════
        // mul_n(even, a, b0)
        // Seeds r0..r7 with (a0,a2,a4,a6)*b0 lo/hi pairs.
        // ════════════════════════════════════════════════════════════════════
        "mul.lo.u32  r0, a0, b0;\n\t"
        "mul.hi.u32  r1, a0, b0;\n\t"
        "mul.lo.u32  r2, a2, b0;\n\t"
        "mul.hi.u32  r3, a2, b0;\n\t"
        "mul.lo.u32  r4, a4, b0;\n\t"
        "mul.hi.u32  r5, a4, b0;\n\t"
        "mul.lo.u32  r6, a6, b0;\n\t"
        "mul.hi.u32  r7, a6, b0;\n\t"

        // ════════════════════════════════════════════════════════════════════
        // mul_n(odd, a+1, b0)
        // Seeds o0..o7 with (a1,a3,a5,a7)*b0 lo/hi pairs.
        // ════════════════════════════════════════════════════════════════════
        "mul.lo.u32  o0, a1, b0;\n\t"
        "mul.hi.u32  o1, a1, b0;\n\t"
        "mul.lo.u32  o2, a3, b0;\n\t"
        "mul.hi.u32  o3, a3, b0;\n\t"
        "mul.lo.u32  o4, a5, b0;\n\t"
        "mul.hi.u32  o5, a5, b0;\n\t"
        "mul.lo.u32  o6, a7, b0;\n\t"
        "mul.hi.u32  o7, a7, b0;\n\t"

        // ════════════════════════════════════════════════════════════════════
        // mad_row(&even[2], &odd[0], a, b1)
        //   odd_ptr  = &even[2]  →  r2..r9
        //   even_ptr = &odd[0]   →  o0..o7
        // ════════════════════════════════════════════════════════════════════
        // Chain A — cmad_n(&even[2], a+1, b1, 6): accumulate into r2..r7
        "mad.lo.cc.u32   r2, a1, b1, r2;\n\t"
        "madc.hi.cc.u32  r3, a1, b1, r3;\n\t"
        "madc.lo.cc.u32  r4, a3, b1, r4;\n\t"
        "madc.hi.cc.u32  r5, a3, b1, r5;\n\t"
        "madc.lo.cc.u32  r6, a5, b1, r6;\n\t"
        "madc.hi.cc.u32  r7, a5, b1, r7;\n\t"
        // trailing pair (odd_ptr[TLC-2], odd_ptr[TLC-1]) = even[8], even[9]
        "madc.lo.cc.u32  r8, a7, b1,  0;\n\t"
        "madc.hi.u32     r9, a7, b1,  0;\n\t"  // ends chain A; does NOT write CC
        // Chain B — cmad_n(&odd[0], a, b1, 8): accumulate into o0..o7
        "mad.lo.cc.u32   o0, a0, b1, o0;\n\t"  // resets CC, starts chain B
        "madc.hi.cc.u32  o1, a0, b1, o1;\n\t"
        "madc.lo.cc.u32  o2, a2, b1, o2;\n\t"
        "madc.hi.cc.u32  o3, a2, b1, o3;\n\t"
        "madc.lo.cc.u32  o4, a4, b1, o4;\n\t"
        "madc.hi.cc.u32  o5, a4, b1, o5;\n\t"
        "madc.lo.cc.u32  o6, a6, b1, o6;\n\t"
        "madc.hi.cc.u32  o7, a6, b1, o7;\n\t"
        // merge carry out of chain B into r9 (odd_ptr[TLC-1] = addc(r9, 0))
        "addc.u32        r9, r9,  0;\n\t"

        // ════════════════════════════════════════════════════════════════════
        // Loop i=2, pass 1: mad_row(&odd[2], &even[2], a, b2)
        //   odd_ptr  = &odd[2]   →  o2..o9
        //   even_ptr = &even[2]  →  r2..r9
        // ════════════════════════════════════════════════════════════════════
        // Chain A — cmad_n(&odd[2], a+1, b2, 6)
        "mad.lo.cc.u32   o2, a1, b2, o2;\n\t"
        "madc.hi.cc.u32  o3, a1, b2, o3;\n\t"
        "madc.lo.cc.u32  o4, a3, b2, o4;\n\t"
        "madc.hi.cc.u32  o5, a3, b2, o5;\n\t"
        "madc.lo.cc.u32  o6, a5, b2, o6;\n\t"
        "madc.hi.cc.u32  o7, a5, b2, o7;\n\t"
        // trailing: o8, o9
        "madc.lo.cc.u32  o8, a7, b2,  0;\n\t"
        "madc.hi.u32     o9, a7, b2,  0;\n\t"
        // Chain B — cmad_n(&even[2], a, b2, 8)
        "mad.lo.cc.u32   r2, a0, b2, r2;\n\t"
        "madc.hi.cc.u32  r3, a0, b2, r3;\n\t"
        "madc.lo.cc.u32  r4, a2, b2, r4;\n\t"
        "madc.hi.cc.u32  r5, a2, b2, r5;\n\t"
        "madc.lo.cc.u32  r6, a4, b2, r6;\n\t"
        "madc.hi.cc.u32  r7, a4, b2, r7;\n\t"
        "madc.lo.cc.u32  r8, a6, b2, r8;\n\t"
        "madc.hi.cc.u32  r9, a6, b2, r9;\n\t"
        "addc.u32        o9, o9,  0;\n\t"

        // ════════════════════════════════════════════════════════════════════
        // Loop i=2, pass 2: mad_row(&even[4], &odd[2], a, b3)
        //   odd_ptr  = &even[4]  →  r4..r11
        //   even_ptr = &odd[2]   →  o2..o9
        // ════════════════════════════════════════════════════════════════════
        // Chain A — cmad_n(&even[4], a+1, b3, 6)
        "mad.lo.cc.u32   r4,  a1, b3, r4;\n\t"
        "madc.hi.cc.u32  r5,  a1, b3, r5;\n\t"
        "madc.lo.cc.u32  r6,  a3, b3, r6;\n\t"
        "madc.hi.cc.u32  r7,  a3, b3, r7;\n\t"
        "madc.lo.cc.u32  r8,  a5, b3, r8;\n\t"
        "madc.hi.cc.u32  r9,  a5, b3, r9;\n\t"
        // trailing: r10, r11
        "madc.lo.cc.u32  r10, a7, b3,  0;\n\t"
        "madc.hi.u32     r11, a7, b3,  0;\n\t"
        // Chain B — cmad_n(&odd[2], a, b3, 8)
        "mad.lo.cc.u32   o2,  a0, b3, o2;\n\t"
        "madc.hi.cc.u32  o3,  a0, b3, o3;\n\t"
        "madc.lo.cc.u32  o4,  a2, b3, o4;\n\t"
        "madc.hi.cc.u32  o5,  a2, b3, o5;\n\t"
        "madc.lo.cc.u32  o6,  a4, b3, o6;\n\t"
        "madc.hi.cc.u32  o7,  a4, b3, o7;\n\t"
        "madc.lo.cc.u32  o8,  a6, b3, o8;\n\t"
        "madc.hi.cc.u32  o9,  a6, b3, o9;\n\t"
        "addc.u32        r11, r11,  0;\n\t"

        // ════════════════════════════════════════════════════════════════════
        // Loop i=4, pass 1: mad_row(&odd[4], &even[4], a, b4)
        //   odd_ptr  = &odd[4]   →  o4..o11
        //   even_ptr = &even[4]  →  r4..r11
        // ════════════════════════════════════════════════════════════════════
        // Chain A — cmad_n(&odd[4], a+1, b4, 6)
        "mad.lo.cc.u32   o4,  a1, b4, o4;\n\t"
        "madc.hi.cc.u32  o5,  a1, b4, o5;\n\t"
        "madc.lo.cc.u32  o6,  a3, b4, o6;\n\t"
        "madc.hi.cc.u32  o7,  a3, b4, o7;\n\t"
        "madc.lo.cc.u32  o8,  a5, b4, o8;\n\t"
        "madc.hi.cc.u32  o9,  a5, b4, o9;\n\t"
        // trailing: o10, o11
        "madc.lo.cc.u32  o10, a7, b4,  0;\n\t"
        "madc.hi.u32     o11, a7, b4,  0;\n\t"
        // Chain B — cmad_n(&even[4], a, b4, 8)
        "mad.lo.cc.u32   r4,  a0, b4,  r4;\n\t"
        "madc.hi.cc.u32  r5,  a0, b4,  r5;\n\t"
        "madc.lo.cc.u32  r6,  a2, b4,  r6;\n\t"
        "madc.hi.cc.u32  r7,  a2, b4,  r7;\n\t"
        "madc.lo.cc.u32  r8,  a4, b4,  r8;\n\t"
        "madc.hi.cc.u32  r9,  a4, b4,  r9;\n\t"
        "madc.lo.cc.u32  r10, a6, b4, r10;\n\t"
        "madc.hi.cc.u32  r11, a6, b4, r11;\n\t"
        "addc.u32        o11, o11,  0;\n\t"

        // ════════════════════════════════════════════════════════════════════
        // Loop i=4, pass 2: mad_row(&even[6], &odd[4], a, b5)
        //   odd_ptr  = &even[6]  →  r6..r13
        //   even_ptr = &odd[4]   →  o4..o11
        // ════════════════════════════════════════════════════════════════════
        // Chain A — cmad_n(&even[6], a+1, b5, 6)
        "mad.lo.cc.u32   r6,  a1, b5,  r6;\n\t"
        "madc.hi.cc.u32  r7,  a1, b5,  r7;\n\t"
        "madc.lo.cc.u32  r8,  a3, b5,  r8;\n\t"
        "madc.hi.cc.u32  r9,  a3, b5,  r9;\n\t"
        "madc.lo.cc.u32  r10, a5, b5, r10;\n\t"
        "madc.hi.cc.u32  r11, a5, b5, r11;\n\t"
        // trailing: r12, r13
        "madc.lo.cc.u32  r12, a7, b5,  0;\n\t"
        "madc.hi.u32     r13, a7, b5,  0;\n\t"
        // Chain B — cmad_n(&odd[4], a, b5, 8)
        "mad.lo.cc.u32   o4,  a0, b5,  o4;\n\t"
        "madc.hi.cc.u32  o5,  a0, b5,  o5;\n\t"
        "madc.lo.cc.u32  o6,  a2, b5,  o6;\n\t"
        "madc.hi.cc.u32  o7,  a2, b5,  o7;\n\t"
        "madc.lo.cc.u32  o8,  a4, b5,  o8;\n\t"
        "madc.hi.cc.u32  o9,  a4, b5,  o9;\n\t"
        "madc.lo.cc.u32  o10, a6, b5, o10;\n\t"
        "madc.hi.cc.u32  o11, a6, b5, o11;\n\t"
        "addc.u32        r13, r13,  0;\n\t"

        // ════════════════════════════════════════════════════════════════════
        // Loop i=6, pass 1: mad_row(&odd[6], &even[6], a, b6)
        //   odd_ptr  = &odd[6]   →  o6..o13
        //   even_ptr = &even[6]  →  r6..r13
        // ════════════════════════════════════════════════════════════════════
        // Chain A — cmad_n(&odd[6], a+1, b6, 6)
        "mad.lo.cc.u32   o6,  a1, b6,  o6;\n\t"
        "madc.hi.cc.u32  o7,  a1, b6,  o7;\n\t"
        "madc.lo.cc.u32  o8,  a3, b6,  o8;\n\t"
        "madc.hi.cc.u32  o9,  a3, b6,  o9;\n\t"
        "madc.lo.cc.u32  o10, a5, b6, o10;\n\t"
        "madc.hi.cc.u32  o11, a5, b6, o11;\n\t"
        // trailing: o12, o13
        "madc.lo.cc.u32  o12, a7, b6,  0;\n\t"
        "madc.hi.u32     o13, a7, b6,  0;\n\t"
        // Chain B — cmad_n(&even[6], a, b6, 8)
        "mad.lo.cc.u32   r6,  a0, b6,  r6;\n\t"
        "madc.hi.cc.u32  r7,  a0, b6,  r7;\n\t"
        "madc.lo.cc.u32  r8,  a2, b6,  r8;\n\t"
        "madc.hi.cc.u32  r9,  a2, b6,  r9;\n\t"
        "madc.lo.cc.u32  r10, a4, b6, r10;\n\t"
        "madc.hi.cc.u32  r11, a4, b6, r11;\n\t"
        "madc.lo.cc.u32  r12, a6, b6, r12;\n\t"
        "madc.hi.cc.u32  r13, a6, b6, r13;\n\t"
        "addc.u32        o13, o13,  0;\n\t"

        // ════════════════════════════════════════════════════════════════════
        // Loop i=6, pass 2: mad_row(&even[8], &odd[6], a, b7)
        //   odd_ptr  = &even[8]  →  r8..r15
        //   even_ptr = &odd[6]   →  o6..o13
        // ════════════════════════════════════════════════════════════════════
        // Chain A — cmad_n(&even[8], a+1, b7, 6)
        "mad.lo.cc.u32   r8,  a1, b7,  r8;\n\t"
        "madc.hi.cc.u32  r9,  a1, b7,  r9;\n\t"
        "madc.lo.cc.u32  r10, a3, b7, r10;\n\t"
        "madc.hi.cc.u32  r11, a3, b7, r11;\n\t"
        "madc.lo.cc.u32  r12, a5, b7, r12;\n\t"
        "madc.hi.cc.u32  r13, a5, b7, r13;\n\t"
        // trailing: r14, r15
        "madc.lo.cc.u32  r14, a7, b7,  0;\n\t"
        "madc.hi.u32     r15, a7, b7,  0;\n\t"
        // Chain B — cmad_n(&odd[6], a, b7, 8)
        "mad.lo.cc.u32   o6,  a0, b7,  o6;\n\t"
        "madc.hi.cc.u32  o7,  a0, b7,  o7;\n\t"
        "madc.lo.cc.u32  o8,  a2, b7,  o8;\n\t"
        "madc.hi.cc.u32  o9,  a2, b7,  o9;\n\t"
        "madc.lo.cc.u32  o10, a4, b7, o10;\n\t"
        "madc.hi.cc.u32  o11, a4, b7, o11;\n\t"
        "madc.lo.cc.u32  o12, a6, b7, o12;\n\t"
        "madc.hi.cc.u32  o13, a6, b7, o13;\n\t"
        "addc.u32        r15, r15,  0;\n\t"

        // ════════════════════════════════════════════════════════════════════
        // Final merge: add odd[0..13] into even[1..14], propagate carry
        //
        // even[1]      += odd[0]        (add_cc)
        // even[2..14]  += odd[1..13]    (addc_cc chain)
        // even[15]     += carry          (addc)
        // ════════════════════════════════════════════════════════════════════
        "add.cc.u32   r1,  r1,  o0;\n\t"
        "addc.cc.u32  r2,  r2,  o1;\n\t"
        "addc.cc.u32  r3,  r3,  o2;\n\t"
        "addc.cc.u32  r4,  r4,  o3;\n\t"
        "addc.cc.u32  r5,  r5,  o4;\n\t"
        "addc.cc.u32  r6,  r6,  o5;\n\t"
        "addc.cc.u32  r7,  r7,  o6;\n\t"
        "addc.cc.u32  r8,  r8,  o7;\n\t"
        "addc.cc.u32  r9,  r9,  o8;\n\t"
        "addc.cc.u32  r10, r10, o9;\n\t"
        "addc.cc.u32  r11, r11, o10;\n\t"
        "addc.cc.u32  r12, r12, o11;\n\t"
        "addc.cc.u32  r13, r13, o12;\n\t"
        "addc.cc.u32  r14, r14, o13;\n\t"
        "addc.u32     r15, r15,  0;\n\t"

        // ── Store outputs ────────────────────────────────────────────────────
        "mov.b32  %0,  r0;   mov.b32  %1,  r1;   mov.b32  %2,  r2;   mov.b32  %3,  r3;\n\t"
        "mov.b32  %4,  r4;   mov.b32  %5,  r5;   mov.b32  %6,  r6;   mov.b32  %7,  r7;\n\t"
        "mov.b32  %8,  r8;   mov.b32  %9,  r9;   mov.b32  %10, r10;  mov.b32  %11, r11;\n\t"
        "mov.b32  %12, r12;  mov.b32  %13, r13;  mov.b32  %14, r14;  mov.b32  %15, r15;\n\t"
        "}"

        : "=r"(rs.limbs[0]),  "=r"(rs.limbs[1]),  "=r"(rs.limbs[2]),  "=r"(rs.limbs[3]),
          "=r"(rs.limbs[4]),  "=r"(rs.limbs[5]),  "=r"(rs.limbs[6]),  "=r"(rs.limbs[7]),
          "=r"(rs.limbs[8]),  "=r"(rs.limbs[9]),  "=r"(rs.limbs[10]), "=r"(rs.limbs[11]),
          "=r"(rs.limbs[12]), "=r"(rs.limbs[13]), "=r"(rs.limbs[14]), "=r"(rs.limbs[15])

        : "r"(as.limbs[0]), "r"(as.limbs[1]), "r"(as.limbs[2]), "r"(as.limbs[3]),
          "r"(as.limbs[4]), "r"(as.limbs[5]), "r"(as.limbs[6]), "r"(as.limbs[7]),
          "r"(bs.limbs[0]), "r"(bs.limbs[1]), "r"(bs.limbs[2]), "r"(bs.limbs[3]),
          "r"(bs.limbs[4]), "r"(bs.limbs[5]), "r"(bs.limbs[6]), "r"(bs.limbs[7])
    );
}


// ─── These functions are unchanged from the original ────────────────────────

static __device__ __forceinline__ void add_limbs_device(const uint32_t *x, const uint32_t *y, uint32_t *r) {
    uint32_t c0, c1, c2, c3, c4, c5, c6, c7;
    asm(
        "{\n\t"
        ".reg .u32 x0,x1,x2,x3,x4,x5,x6,x7;\n\t"
        ".reg .u32 y0,y1,y2,y3,y4,y5,y6,y7;\n\t"
        "mov.b32 x0,%8;  mov.b32 x1,%9;  mov.b32 x2,%10; mov.b32 x3,%11;\n\t"
        "mov.b32 x4,%12; mov.b32 x5,%13; mov.b32 x6,%14; mov.b32 x7,%15;\n\t"
        "mov.b32 y0,%16; mov.b32 y1,%17; mov.b32 y2,%18; mov.b32 y3,%19;\n\t"
        "mov.b32 y4,%20; mov.b32 y5,%21; mov.b32 y6,%22; mov.b32 y7,%23;\n\t"
        "add.cc.u32   %0, x0, y0;\n\t"
        "addc.cc.u32  %1, x1, y1;\n\t"
        "addc.cc.u32  %2, x2, y2;\n\t"
        "addc.cc.u32  %3, x3, y3;\n\t"
        "addc.cc.u32  %4, x4, y4;\n\t"
        "addc.cc.u32  %5, x5, y5;\n\t"
        "addc.cc.u32  %6, x6, y6;\n\t"
        "addc.u32     %7, x7, y7;\n\t"
        "}"
        : "=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3),"=r"(c4),"=r"(c5),"=r"(c6),"=r"(c7)
        : "r"(x[0]),"r"(x[1]),"r"(x[2]),"r"(x[3]),"r"(x[4]),"r"(x[5]),"r"(x[6]),"r"(x[7]),
          "r"(y[0]),"r"(y[1]),"r"(y[2]),"r"(y[3]),"r"(y[4]),"r"(y[5]),"r"(y[6]),"r"(y[7])
    );
    r[0]=c0; r[1]=c1; r[2]=c2; r[3]=c3; r[4]=c4; r[5]=c5; r[6]=c6; r[7]=c7;
}

bigint __device__ __forceinline__ get_256_bit_result(const bigint_wide &xs) {
    const uint32_t *x = xs.limbs;
    bigint out{};
    add_limbs_device(x, &x[TLC], out.limbs);
    return out;
}


// ─── Kernel and host entry points (unchanged) ────────────────────────────────

template <int N>
__global__ void multVectorsKernel(bigint *in1, const bigint *in2, bigint_wide *out, size_t n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
    {
        bigint i1 = in1[tid];
        const bigint i2 = in2[tid];
        bigint_wide o = {0};
        for (int i = 0; i < N - 1; i++) {
            multiply_raw_device(i1, i2, o);
            i1 = get_256_bit_result(o);
        }
        multiply_raw_device(i1, i2, out[tid]);
    }
}

template <int N>
int mult_vectors(bigint in1[], const bigint in2[], bigint_wide *out, size_t n)
{
    int threads_per_block = 128;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block + 1;
    multVectorsKernel<N><<<num_blocks, threads_per_block>>>(in1, in2, out, n);
    return 0;
}

extern "C"
int multiply_test(bigint in1[], const bigint in2[], bigint_wide *out, size_t n)
{
    try {
        mult_vectors<1>(in1, in2, out, n);
        return CUDA_SUCCESS;
    } catch (const std::runtime_error &ex) {
        return -1;
    }
}

extern "C"
int multiply_bench(bigint in1[], const bigint in2[], bigint_wide *out, size_t n)
{
    try {
        mult_vectors<500>(in1, in2, out, n);
        return CUDA_SUCCESS;
    } catch (const std::runtime_error &ex) {
        return -1;
    }
}
