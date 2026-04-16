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

### Nvidia Speed of Light Profile Stats

#### GPU Speed Of Light Throughput

| Metric Name             | Unit  | mult-32b-limbs-one-asm-block | mult-64b-limbs   | mult-32b-limbs |
| :---------------------- | :---- | :--------------------------- | :--------------- | :------------- |
| Duration                | ms    | 595.80                       | 843.05           | 549.63         |
| Elapsed Cycles          | cycle | 902,408,579                  | 1,276,753,856    | 832,405,586    |
| SM Active Cycles        | cycle | 906,461,022.74               | 1,273,128,099.35 | 830,105,680.80 |
| Compute (SM) Throughput | %     | 71.31                        | 84.14            | 78.27          |
| Memory Throughput       | %     | 26.63                        | 2.50             | 6.68           |
| DRAM Throughput         | %     | 1.50                         | 1.09             | 1.61           |
| L1/TEX Cache Throughput | %     | 0.60                         | 1.73             | 6.68           |
| L2 Cache Throughput     | %     | 26.55                        | 2.50             | 3.74           |
| DRAM Frequency          | Ghz   | 13.99                        | 13.99            | 13.99          |
| SM Frequency            | Ghz   | 1.51                         | 1.51             | 1.51           |

---

#### Compute Workload Analysis

| Metric Name          | Unit       | mult-32b-limbs-one-asm-block | mult-64b-limbs | mult-32b-limbs |
| :------------------- | :--------- | :--------------------------- | :------------- | :------------- |
| SM Busy              | %          | 71.31                        | 84.14          | 78.27          |
| Issue Slots Busy     | %          | 30.04                        | 40.87          | 33.62          |
| Executed Ipc Active  | inst/cycle | 1.20                         | 1.63           | 1.34           |
| Executed Ipc Elapsed | inst/cycle | 1.20                         | 1.63           | 1.34           |
| Issued Ipc Active    | inst/cycle | 1.20                         | 1.63           | 1.34           |

---

#### Memory Workload Analysis

| Metric Name                  | Unit    | mult-32b-limbs-one-asm-block | mult-64b-limbs | mult-32b-limbs |
| :--------------------------- | :------ | :--------------------------- | :------------- | :------------- |
| Memory Throughput            | Gbyte/s | 7.79                         | 5.65           | 8.41           |
| Mem Busy                     | %       | 26.55                        | 1.43           | 6.68           |
| Max Bandwidth                | %       | 26.63                        | 2.50           | 3.74           |
| L1/TEX Hit Rate              | %       | 62.47                        | 93.02          | 97.97          |
| L2 Hit Rate                  | %       | 101.81                       | 78.93          | 79.24          |
| Mem Pipes Busy               | %       | 0.05                         | 0.16           | 0.82           |
| L2 Persisting Size           | Mbyte   | 7.08                         | 7.08           | 7.08           |
| Local Mem Spilling Requests  | count   | 0                            | 0              | 0              |
| Local Mem Spilling Overhead  | %       | 0                            | 0              | 0              |
| L2 Sector Promotion Misses   | %       | 0                            | 0              | 0              |
| Shared Mem Spilling Requests | count   | 0                            | 0              | 0              |
| Shared Mem Spilling Overhead | %       | 0                            | 0              | 0              |
| L2 Compression Success Rate  | %       | 0                            | 0              | 0              |
| L2 Compression Ratio         | ratio   | 0                            | 0              | 0              |
| L2 Compression Input Sectors | sector  | 0                            | 0              | 0              |

---

#### Scheduler Statistics

| Metric Name                | Unit  | mult-32b-limbs-one-asm-block | mult-64b-limbs | mult-32b-limbs |
| :------------------------- | :---- | :--------------------------- | :------------- | :------------- |
| One or More Eligible       | %     | 29.94                        | 40.87          | 33.62          |
| No Eligible                | %     | 70.06                        | 59.13          | 66.38          |
| Active Warps Per Scheduler | warp  | 9.66                         | 8.99           | 9.92           |
| Eligible Warps Per Sched.  | warp  | 1.06                         | 2.89           | 2.44           |
| Issued Warp Per Scheduler  | count | 0.30                         | 0.41           | 0.34           |

