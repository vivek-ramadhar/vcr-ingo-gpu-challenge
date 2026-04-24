# Ingo GPU challenge

Integer arithmetic is the backbone of cryptographic algorithms. However, the typical widths of numbers that provide acceptable security are not directly supported by standard hardware such as CPUs and GPUs. Therefore, we need to represent our integers as arrays of multiple "limbs", either 32 or 64 bits in size.

## The challenge

Design a kernel that multiplies pairs of 256-bit numbers, getting 512-bit results. The goal is to maximize the throughput of the mutliplier.

## Testing suite

To benchmark and test the code, we use a Rust wrapper. For Rust installation, see [this](https://www.rust-lang.org/tools/install) link.

Once Rust is installed, the correctness of the code can be verified by running:

```
cargo test
```

And the performance can be measured by running:

```
cargo bench
```

There is baseline code that implements section 4 of [this](http://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf) paper in `cuda/mult.cu` file. It is expected that only this file will be edited, the rest of the repo is meant to provide the infrastructure for easy testing and benchmarking.

You can parallelize the multiplier, or do the opposite - the only optimization goal is throughput. You can also change between 32-bit and 64-bit limbs, try using floating point arithmetic etc.

Any machine can be used, the only limitation is using a single GPU for measurements.

If there are any questions, you can file an issue in this repository.

Good luck and have fun!

## Vivek's Analysis

Machine: ROG Strix G16 G614FR - AMD Ryzen 9 9955HX3D 16-Core Processor - GeForce RTX 5070ti Laptop GPU - Cuda Capability: sm_120

### 32-Bit Limbs, 64-bit Limbs, and Karatsuba Multiplication

The [challenge repo](https://github.com/ingonyama-zk/ingo-gpu-challenge/tree/main) starts by claiming it implements section 4 of [Optimizing Modular Multiplication for NVIDIA's Maxwell GPUs by Emmart et al.](https://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf) in `cuda/mult.cu`. But `cuda/mult.cu` seems to actually implement a schoolbook multiplication of two 256-bit integers with 8 32-bit words, or as is called within Emmart et al., the Row Oriented Approach.

<details>
    <summary>Aside on Section 4 and Nvidia GPU Multiplier Architecture</summary>
    The paper targets Maxwell architecture and compute capability 5.x which lacks a hardware 32-bit, so it can't do simple multiplication with 32-bit words and breaks the 32-bit multiplier into two 16-bit halves. *Section IV. Two-Pass Algorithms For Multiplication and Montgomery Reduction* focuses on how to efficiently mulitply a 128-bit integer with 4 32-bit words by a 64-bit integer with 2 32-bit words, using 16-bit products. However, GPU technology has evolved significantly since then; cc 5.x predates the sm_## convention! Volta architecture (cc 7.x) introduced separate INT32 and FP32 pipelines ([Table 4.1 from Jia et al.](https://arxiv.org/pdf/1804.06826)), enabling hardware multiplies with 32-bit integers. The tradeoff is worse hardware utilization when using primarily one type (mostly INT32 or mostly FP32). Nvidia tried to solve this issue with Ampere (sm_80) and the GA10x SM ([Ampere Whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf)). Instead of a unique datapath for each type, they introduced one datapath that just does FP32 ops and another datapath that can do FP32 and INT32 ops. This enabled a GA10x to do either 32 FP32 ops per clock or 16FP32 and 16INT32 ops per clock. Blackwell (sm_100/sm_120) took Ampere's approach to the next level by making *all* cores handle either INT32 or FP32 ops ([Blackwell Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)). So with Blackwell's GB202 SMs, they unified the datapath, effectively doubling the maximum int operations in the process.

    Summarizing the evolution of the device level integer multiplier over time:
        - Telsa     (1.x)  : IMAD24
        - Fermi     (2.x)  : IMAD32
        - Kepler    (3.x)  : IMAD32
        - Maxwell   (5.x)  : XMAD
        - Pascal    (6.x)  : XMAD
        - Volta     (7.x)  : IMAD32     [Dedicated INT and FP pipelines]
        - Ampere    (8.x)  : IMAD32     [Two datapaths, one just for INTs and one INTs/FPs]
        - Hopper    (9.x)  : IMAD32     [Dedicated INT and FP pipelines]
        - Blackwell (10.x) : IMAD32     [Unified datapath, all cores are INT/FP]


    Returning to the task of multiplying multi-precision integers of 32-bit words, it seems like the approach in *Section IV* was highly specific to the Maxwell and Pascal generation of hardware. Almost every other generation of GPU uses the IMAD32 multiplier, so the approach outlined in Section IV is sub-optimal for any of the most recent GPU architectures going back to Volta in ~2017/2018. This is probably why cuda/mult.cu implements the schoolbook multiplication rather than the Two-Pass Montgomery Multiply, since it can take advantage of native 32-bit multipliers instructions that have been around since 2017/2018 and doesn't need to split 32-bit words into 16-bit halves. The schoolbook multiply approach is actually adressed in Emmart et al. within Section II, they call it the Row Oriented Approach. So to be pedantic, it would probably be more accurate to say `cuda/mult.cu` implements Section II of the paper rather than Section IV of the paper.

</details>

### Bug Report

_Disclaimer: I don't write or use Rust. I only installed cargo and Rust to run these benchmarks._

While testing an iteration of the kernel, I got a weird error after doing `cargo clean` and attempting `cargo bench`:

```
running 1 test
test test_mult ... ignored

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running benches\multiplication_benchmark.rs (target\release\deps\mult-bb412473bdb90181.exe)

thread 'main' (43848) panicked at benches\multiplication_benchmark.rs:20:57:
called `Result::unwrap()` on an `Err` value: [1991065059, 3609320981, 359871887, 1916112490, 623082313, 3606681219, 3934619629]
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
error: bench failed, to rerun pass `--bench mult`
```

Retrying `cargo bench` on the same cached build worked and then doing `cargo clean` and `cargo bench` again passed fine so it seemed to indicate it's non-deterministic.

The actual error code points to 20 in the multiplciation_benchmark.rs:

```
BigInt256 { s: b_num.to_u32_digits().try_into().unwrap() }
```

Dropping this into Claude, it points out this is inconsistent with `src/lib.rs`:

```
// lib.rs test — always had the resize, always correct
let mut a_digits = a_num.to_u32_digits();
a_digits.resize(TLC, 0);
BigInt256 { s: a_digits.try_into().unwrap() }

// multiplication_benchmark.rs — never had the resize, always broken
BigInt256 { s: a_num.to_u32_digits().try_into().unwrap() }
```

The content of multiplication_benchmark.rs seems to be using `sample_random_bigints256()` to generate inputs for `multiply_cuda`, which calls the multiply kernel. Looking at `src/lib.rs`, `sample_random_bigints256()` seems to be using `RandomBits::new(256)` to create two random 256-bit integer. With that additional context, it seems clear what happened that. Line 20 tried to convert the random 256-bit integer into its u32_digits and unwrap and then convert to a BigInt256, but the random integer only had 7 u32 digits, not the 8 necessary for the type conversion. `src/lib.rs` shows the bug fix: set `a_digits = a_num.to_u32_digits()` and then resize to the correct limb count, `a_digits.resize(TLC, 0)`, and then finally cast to BigInt256, `BigInt256 { s : a_digits.try_into().unwrap() }`.

For fun, lets calculate how likely it is that `RandomBits::new(256)` generates a number that has less than 8 32-bit digits. It generates a random number over the range [0, 2^256-1]. And we are looking for numbers with less than or equal to 7 32-bit digits, or numbers with the 8th 32-bit word is exactly 0. This would be numbers in the range [0, 2^(256-32) - 1 = 2^224 - 1]. Thus, the probability of generating a number with the 8th 32-bit digit 0 is 2^224 / 2^256 = 1/2^32.

From one of the benchmarks that worked: `Benchmarking Benchmarking multiplication of size 33554432: Warming up for 3.0000 s`, so the probability of getting this error `cargo bench` is 2\*33554432 \* (1/2^32) = 0.015625. Thus, the error has a 1.5625% chance of occurring. Would be pretty nasty bug if it wasn't such a small codebase and easy to fix!

The inconsistency with other Rust code in the repo, the non-deterministic result, and the fact it was a Rust host-code error and not a CUDA code error leads me to believe it is a bug with the benchmark code and not an issue with my changes to mult.cu or build.rs

### Profiling

#### Nsight Dump

Each profiled with `ncu --set full .\*.exe`

mult-32b-limbs.exe
Duration: 710.09ms
Compute (SM) Throughput - 72.27%
Issued Instructions inst 51,350,268,286
issues an instruction every 3.0 cycles
On average, each warp of this workload spends 17.6 cycles being stalled waiting for the execution pipe to be available.
Warp Cycles Per Issued Instruction cycle 29.47

Section: Scheduler Statistics

---

Metric Name Metric Unit Metric Value

---

One or More Eligible % 33.61
Issued Warp Per Scheduler 0.34
No Eligible % 66.39
Active Warps Per Scheduler warp 9.91
Eligible Warps Per Scheduler warp 2.43

---

OPT Est. Local Speedup: 27.73%  
 Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only  
 issues an instruction every 3.0 cycles. This might leave hardware resources underutilized and may lead to  
 less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average  
 of 9.91 active warps per scheduler, but only an average of 2.43 warps were eligible per cycle. Eligible  
 warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no  
 eligible warp results in no instruction being issued and the issue slot remains unused. To increase the  
 number of eligible warps, avoid possible load imbalances due to highly different execution durations per  
 warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too.

Section: Warp State Statistics

---

Metric Name Metric Unit Metric Value

---

Warp Cycles Per Issued Instruction cycle 29.47
Warp Cycles Per Executed Instruction cycle 29.47
Avg. Active Threads Per Warp 32
Avg. Not Predicated Off Threads Per Warp 32.00

---

OPT Est. Speedup: 27.73%  
 On average, each warp of this workload spends 17.6 cycles being stalled waiting for the execution pipe to be  
 available. This stall occurs when all active warps execute their next instruction on a specific,  
 oversubscribed math pipeline. Try to increase the number of active warps to hide the existent latency or try  
 changing the instruction mix to utilize all available pipelines in a more balanced way. This stall type  
 represents about 59.6% of the total average of 29.5 cycles between issuing two instructions.

---

mult-32b-limbs-karatsuba.exe
Duration: 734.19ms
Compute (SM) Throughput 75.97%
Issued Instructions inst 75,650,975,126
Issues an instruction every 2.0 cycles
On average, each warp of this workload spends 10.0 cycles being stalled waiting for the execution pipe to be available.
Warp Cycles Per Issued Instruction cycle 18.02

Section: Scheduler Statistics

---

Metric Name Metric Unit Metric Value

---

One or More Eligible % 49.64
Issued Warp Per Scheduler 0.50
No Eligible % 50.36
Active Warps Per Scheduler warp 8.94
Eligible Warps Per Scheduler warp 2.80

---

OPT Est. Local Speedup: 24.03%  
 Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only  
 issues an instruction every 2.0 cycles. This might leave hardware resources underutilized and may lead to  
 less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average  
 of 8.94 active warps per scheduler, but only an average of 2.80 warps were eligible per cycle. Eligible  
 warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no  
 eligible warp results in no instruction being issued and the issue slot remains unused. To increase the  
 number of eligible warps, avoid possible load imbalances due to highly different execution durations per  
 warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too.

Section: Warp State Statistics

---

Metric Name Metric Unit Metric Value

---

Warp Cycles Per Issued Instruction cycle 18.02
Warp Cycles Per Executed Instruction cycle 18.02
Avg. Active Threads Per Warp 32
Avg. Not Predicated Off Threads Per Warp 32.00

---

OPT Est. Speedup: 24.03%  
 On average, each warp of this workload spends 10.0 cycles being stalled waiting for the execution pipe to be  
 available. This stall occurs when all active warps execute their next instruction on a specific,  
 oversubscribed math pipeline. Try to increase the number of active warps to hide the existent latency or try  
 changing the instruction mix to utilize all available pipelines in a more balanced way. This stall type  
 represents about 55.3% of the total average of 18.0 cycles between issuing two instructions.

---

mult-64b-limbs.exe
Duration: 1100ms
Compute (SM) Throughput 78.02%
Issued Instructions inst 95,745,210,944
issues an instruction every 2.4 cycles
this workload spends 13.0 cycles being stalled waiting for the execution pipe
Warp Cycles Per Issued Instruction cycle 21.99

Section: Scheduler Statistics

---

Metric Name Metric Unit Metric Value

---

One or More Eligible % 40.87
Issued Warp Per Scheduler 0.41
No Eligible % 59.13
Active Warps Per Scheduler warp 8.99
Eligible Warps Per Scheduler warp 2.89

---

OPT Est. Local Speedup: 21.98%
Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only
issues an instruction every 2.4 cycles. This might leave hardware resources underutilized and may lead to
less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average
of 8.99 active warps per scheduler, but only an average of 2.89 warps were eligible per cycle. Eligible
warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no
eligible warp results in no instruction being issued and the issue slot remains unused. To increase the
number of eligible warps, avoid possible load imbalances due to highly different execution durations per
warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too.

Section: Warp State Statistics

---

Metric Name Metric Unit Metric Value

---

Warp Cycles Per Issued Instruction cycle 21.99
Warp Cycles Per Executed Instruction cycle 21.99
Avg. Active Threads Per Warp 32
Avg. Not Predicated Off Threads Per Warp 32.00

---

OPT Est. Speedup: 21.98%
On average, each warp of this workload spends 13.0 cycles being stalled waiting for the execution pipe to be
available. This stall occurs when all active warps execute their next instruction on a specific,
oversubscribed math pipeline. Try to increase the number of active warps to hide the existent latency or try
changing the instruction mix to utilize all available pipelines in a more balanced way. This stall type
represents about 59.0% of the total average of 22.0 cycles between issuing two instructions.

---

#### Key Stats:

---

## mult-32b-limbs.exe

Duration: 710.09ms
Compute (SM) Throughput - 72.27%
Issued Instructions inst 51,350,268,286
issues an instruction every 3.0 cycles
On average, each warp of this workload spends 17.6 cycles being stalled waiting for the execution pipe to be available.
Warp Cycles Per Issued Instruction cycle 29.47

---

## mult-64b-limbs.exe

Duration: 1100ms
Compute (SM) Throughput 78.02%
Issued Instructions inst 95,745,210,944
issues an instruction every 2.4 cycles
On average, each warp of this workload spends 13.0 cycles being stalled waiting for the execution pipe to be available.
Warp Cycles Per Issued Instruction cycle 21.99

---

## mult-32b-limbs-karatsuba.exe

Duration: 734.19ms
Compute (SM) Throughput 75.97%
Issued Instructions inst 75,650,975,126
Issues an instruction every 2.0 cycles
On average, each warp of this workload spends 10.0 cycles being stalled waiting for the execution pipe to be available.
Warp Cycles Per Issued Instruction cycle 18.02

---
