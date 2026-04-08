# High Performance Boson Sampling Simulation via Data-Flow Engines

[![DOI](https://img.shields.io/badge/DOI-10.1088%2F1367--2630%2Fad313b-blue)](https://doi.org/10.1088/1367-2630/ad313b)
[![Journal](https://img.shields.io/badge/Journal-New%20Journal%20of%20Physics-green)](https://iopscience.iop.org/journal/1367-2630)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

**Code supplement** for the paper:

> Gregory Morse, Tomasz Rybotycki, Ágoston Kaposi, Zoltán Kolarovszki, Uroš Stojčić, Tamás Kozsik, Oskar Mencer, Michał Oszmaniec, Zoltán Zimborás and Péter Rakyta,
> *"High performance Boson sampling simulation via data-flow engines"*,
> **New Journal of Physics**, Vol. 26, 033033, March 2024.
> https://doi.org/10.1088/1367-2630/ad313b

The companion code supplement (CPU/GPU C++ Python-extension engines for the Piquasso framework) is available at: [piquassoboost](https://github.com/Budapest-Quantum-Computing-Group/piquassoboost).

---

## Table of Contents

- [Abstract](#abstract)
- [Repository Overview](#repository-overview)
- [Repository Structure](#repository-structure)
- [Algorithms](#algorithms)
- [Hardware and Software Requirements](#hardware-and-software-requirements)
- [Build Instructions](#build-instructions)
- [Running the Benchmarks](#running-the-benchmarks)
- [Performance Highlights](#performance-highlights)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Citation](#citation)

---

## Abstract

Boson sampling (BS) is viewed as an accessible quantum computing paradigm to demonstrate computational advantage compared to classical computers. The simulation of BS experiments requires the evaluation of a vast number of matrix permanents. This repository implements a novel generalization of the Balasubramanian–Bax–Franklin–Glynn (BB/FG) permanent formula that efficiently incorporates photon occupation multiplicities via *n*-ary Gray code ordering. The algorithm is deployed on **FPGA-based Data-Flow Engines (DFEs)** — specifically Xilinx Alveo U250 cards programmed using Maxeler Technologies' MaxCompiler — to accelerate both ideal and lossy boson sampling simulations up to **40 photons**. Drawing samples from a 60-mode interferometer, the achieved rate averages around **80 s per sample** using 4 FPGA chips, representing a substantial speedup over CPU implementations requiring quad-precision arithmetic.

---

## Repository Overview

This repository contains three main algorithmic components implemented for FPGA-based DFE hardware (Maxeler/Xilinx Alveo U250) and their CPU reference implementations, as well as GPU research code targeting the Groq Tensor Streaming Processor (TSP):

| Component | Hardware Target | Algorithm |
|---|---|---|
| `PermanentGlynn/` | Maxeler DFE (Xilinx Alveo U250) | Glynn / BB/FG permanent formula |
| `ZOnePermanent/` | Maxeler DFE (Xilinx Alveo U250) | 0-1 (Fock-state) permanent via exact integer arithmetic |
| `LoopHafnian/` | Groq Tensor Streaming Processor (TSP) | Loop hafnian / matrix-vector Gray-code multiplication |
| `piquassoboost/` | CPU (AMD EPYC, TBB) + DFE | Python C++ extension wrappers for all backends |

The Python bindings in `piquassoboost/` integrate these engines into the [Piquasso](https://github.com/Budapest-Quantum-Computing-Group/piquasso) photonic quantum computing framework.

---

## Repository Structure

```
maxeler/
├── README.md
├── build_instructions.txt           # Full step-by-step build and deployment guide
│
├── permanent_benchmark.py           # Benchmark: Glynn/BBFG permanent (CPU + DFE variants)
├── boson_sampling_benchmark.py      # Benchmark: Full boson sampling simulation (Clifford & Clifford)
├── loop_hafnian_benchmark.py        # Benchmark: Loop hafnian (CPU, long-double, MPFR)
├── zonepermanent_benchmark.py       # Benchmark: 0-1 permanent (CPU reference + DFE)
│
├── PermanentGlynn/                  # Glynn permanent on Maxeler DFE
│   ├── Makefile                     # Master build for SIM/DFE/CPU targets
│   ├── PermanentGlynnDFE/           # MaxJ (Java) kernel sources for FPGA
│   │   └── src/permanentglynn/
│   │       ├── InitializeColSumDFEKernel.maxj   # Column-sum init + Gray-code counter
│   │       ├── PermanentGlynnDFEKernel.maxj     # Complex product-tree reduction kernel
│   │       ├── SumUpPermDFEKernel.maxj          # Permanent accumulator kernel
│   │       ├── PermanentGlynnDFEManager.maxj    # Dataflow graph wiring
│   │       ├── PermanentGlynnDFEManagerAlveoU250.maxj  # Alveo U250 top-level manager
│   │       └── Utility.maxj                     # SLR routing, Gray code, DSP helpers
│   ├── PermanentGlynnCPU/           # C interface library (links MaxSLiCe runtime)
│   ├── PermRepGlynnDFE/             # Repeated-row (boson sampling) variant for DFE
│   ├── PermRepGlynnCPU/             # CPU interface for PermRepGlynn
│   ├── PermanentTestDFE/            # Arithmetic test kernel (fixed-point validation)
│   ├── PermanentTestCPU/            # CPU driver for arithmetic tests
│   └── MaxCompilerFix/              # Patched MaxCompiler library classes
│
├── ZOnePermanent/                   # 0-1 permanent on Maxeler DFE (exact integer arithmetic)
│   ├── Makefile
│   ├── PermanentZOneDFE/            # MaxJ kernel sources
│   │   └── src/permanentzone/
│   │       ├── PermanentZOneDFEKernel.maxj      # Core integer column-sum kernel
│   │       ├── ProdColsDFEKernel.maxj           # Big-integer product reduction
│   │       ├── FinalSumUpPermDFEKernel.maxj     # Final big-integer accumulation
│   │       └── PermanentZOneDFEManagerAlveoU250.maxj  # Top-level manager
│   └── PermanentZOneCPU/            # C interface library
│
├── LoopHafnian/                     # Loop hafnian / unitary decomposition on Groq TSP
│   ├── Makefile
│   ├── graycode.py                  # Groq TSP Gray-code state-machine component
│   ├── vecmat.py                    # Groq TSP matrix-vector multiplication
│   ├── vecmatgcode.py               # Fixed-point Gray-code ordered mat-vec on Groq
│   ├── gsquander.py                 # Groq TSP quantum gate decomposition (SQUANDER)
│   ├── groq_us.c                    # C++ Groq driver (IOP, IOBuffer, multi-device)
│   ├── dumpgroqdoc.py               # Groq API documentation generator
│   └── Groq Chip Vector Matrix Multiplication Demo.ipynb
│
└── piquassoboost/                   # Python C++ extension wrappers (Piquasso integration)
    └── sampling/
        ├── Boson_Sampling_Utilities.py          # Python wrapper classes for all backends
        ├── GlynnPermanentCalculator_Wrapper.hpp # C extension: backend dispatch (lib=0..11)
        ├── ZOnePermanentCalculator_Wrapper.hpp  # C extension: 0-1 permanent (dlopen DFE)
        ├── ChinHuhPermanentCalculator_Wrapper.hpp
        ├── simulation_strategies/
        │   └── GeneralizedCliffordsSimulationStrategy.py  # Clifford & Clifford sampler
        └── source/
            ├── BBFGPermanentCalculator.{h,cpp}            # CPU BBFG double/long-double
            ├── BBFGPermanentCalculatorRepeated.{h,cpp}    # BBFG with multiplicities
            ├── GlynnPermanentCalculatorDFE.{h,cpp}        # DFE interface (single/dual)
            ├── GlynnPermanentCalculatorRepeatedDFE.{h,cpp}
            ├── GlynnPermanentCalculatorInf.{h,cpp}        # MPFR arbitrary precision
            ├── GlynnPermanentCalculatorRepeatedInf.{h,cpp}
            ├── ZOnePermanentCalculator.{h,cpp}            # TBB-parallel 0-1 permanent
            ├── ZOnePermanentCalculatorDFE.{h,cpp}         # DFE 0-1 permanent interface
            └── InfinitePrecisionComplex.h                 # GNU MPFR complex arithmetic
```

---

## Algorithms

### Permanent Calculation

The **matrix permanent** of an *n × n* matrix $A$ is defined as:

$$\text{perm}(A) = \sum_{\sigma \in S_n} \prod_{i=1}^{n} a_{i,\sigma(i)}$$

This repository implements multiple algorithms for evaluating the permanent, each targeting different precision and performance trade-offs:

| Algorithm | Formula | Complexity | Precision | Hardware |
|---|---|---|---|---|
| Ryser (reference) | Ryser | $O(n \cdot 2^n)$ | integer / double | CPU |
| Glynn / BB/FG (Gray-coded) | BB/FG | $O(n \cdot 2^{n-1})$ | double, long-double, MPFR | CPU (TBB multi-thread) |
| Glynn / BB/FG generalized | BB/FG + *n*-ary Gray code | reduced (see §2.2 of paper) | 64–256 bit fixed-point | Single / Dual DFE |
| BBFG with multiplicities | BB/FG + row/col repeats | $O\!\left(\prod_k (M_k+1)/2\right)$ | double, long-double | CPU |
| 0-1 permanent | Glynn / Ryser (integer) | $O(n \cdot 2^n)$ | exact big-integer | CPU (TBB), Single / Dual DFE |

### DFE Architecture (Xilinx Alveo U250, 330 MHz)

The DFE permanent engine consists of four pipelined kernel stages spread across the four Super Logic Regions (SLRs) of the Alveo U250:

1. **`InitializeColSumDFEKernel`** — Receives the input matrix over PCIe, initializes column sums, drives a Gray-code counter with 4× multiplexed streams.
2. **`PermanentGlynnDFEKernel`** — Computes the complex product of column sums via a binary-tree reduction (Karatsuba-style DSP tiling).
3. **`SumUpPermDFEKernel`** — Accumulates Gray-code addends into the final permanent using a single-tick feedback loop.

Fixed-point precision grows from **64-bit input** through intermediate widths up to a **256-bit accumulator**, providing numerical accuracy matching CPU extended-precision (long double) for matrices up to size 40×40.

### Loop Hafnian

The loop hafnian is implemented via the power-trace method (polynomial complexity). CPU reference implementations at double, long-double, and arbitrary (MPFR) precision are provided. An experimental Groq TSP implementation of Gray-code ordered matrix-vector multiplication is also included.

---

## Hardware and Software Requirements

### DFE (Maxeler / Xilinx Alveo U250)

| Requirement | Details |
|---|---|
| FPGA card | Xilinx Alveo U250 |
| FPGA toolchain | Maxeler MaxCompiler 2021.1, Vivado (bundled) |
| Runtime library | MaxSLiCe (Maxeler SLiC Engine runtime) |
| Host CPU | AMD EPYC 7542 (or compatible x86-64 server) |
| PCIe | Gen 3.0 × 16 |
| Operating System | Linux (tested on CentOS / RHEL) |

> **Note:** The `MaxCompilerFix/` directory contains patched MaxCompiler library classes (`com.maxeler.*`) that must be applied to the MaxCompiler installation before building. See [Build Instructions](#build-instructions).

### Groq TSP (LoopHafnian)

| Requirement | Details |
|---|---|
| Hardware | Groq GroqChip processor |
| SDK | Groq SDK (with `groq.api`, `groq.runtime.driver`) |
| Compiler | Groq compiler (`groq.tools.compiler`) |

### CPU / Software

| Requirement | Details |
|---|---|
| C/C++ | GCC ≥ 9 or Intel ICC |
| Python | Python ≥ 3.8 |
| Threading | Intel TBB (Threading Building Blocks) |
| Arbitrary precision | GNU MPFR library |
| Python packages | `piquasso`, `numpy`, `thewalrus` (for benchmarks) |
| Build tools | Apache Ant (for MaxJ), GNU Make |

---

## Build Instructions

> Full details are in [`build_instructions.txt`](build_instructions.txt). This section provides a high-level guide.

### Step 1: Set Up the Workspace

Copy the repository files into the Maxeler workspace and `piquassoboost` directories:

```bash
# DFE kernel sources → Maxeler workspace
cp -r PermanentGlynn/PermanentGlynnDFE/*   ~/workspace/PermanentGlynnDFE/
cp -r PermanentGlynn/PermanentGlynnCPU/*   ~/workspace/PermanentGlynnCPU/
cp    PermanentGlynn/Makefile               ~/workspace/Makefile

# Python extension sources → piquassoboost package
cp -r piquassoboost/sampling/*              ~/piquassoboost/sampling/
cp -r piquassoboost/sampling/source/*       ~/piquassoboost/sampling/source/
cp -r piquassoboost/sampling/simulation_strategies/* \
      ~/piquassoboost/sampling/simulation_strategies/source/

# Optional: benchmark scripts
cp permanent_benchmark.py       ~/piquassoboost/
cp boson_sampling_benchmark.py  ~/piquassoboost/
```

### Step 2: Apply MaxCompiler Patches

```bash
# Copy patched classes into your MaxCompiler installation
cp -r PermanentGlynn/MaxCompilerFix/com  $MAXCOMPILERDIR/lib/com
```

### Step 3: Adjust and Build the piquassoboost C++ Extension

Edit the CMake/setup configuration of piquassoboost to add MPFR and DFE library paths, then:

```bash
cd ~/piquassoboost
python3 setup.py build_ext
cd ..
```

### Step 4: Build FPGA Bitstreams (or Simulation Model)

```bash
cd ~/workspace

# Software simulation (fast, no hardware needed):
make SIM

# Hardware bitstream (requires MaxCompiler + Vivado, takes ~hours):
make DFE

# Optional variants:
make SIMDUAL    # Dual-DFE simulation
make DFEDUAL    # Dual-DFE hardware bitstream
make SIMF       # Float-mode simulation
make DFEF       # Float-mode hardware bitstream
make SIMREP     # Repeated-permanent simulation
make DFEREP     # Repeated-permanent hardware bitstream
```

### Step 5: Run in Simulation Mode

```bash
cd ~/workspace
make CPU        # Build CPU interface library and run with simulation backend
```

### Step 6: Run on DFE Hardware

```bash
cd ~/piquassoboost
env LD_LIBRARY_PATH=/home/$USER/workspace/PermanentGlynnCPU/dist/release/lib:$LD_LIBRARY_PATH \
    python3 permanent_benchmark.py
```

> **Simulation vs. Hardware Detection:** The environment variable `SLIC_CONF` controls whether the Maxeler runtime uses the software simulation or real FPGA hardware. Its presence triggers simulation mode; running without it targets the DFE.

---

## Running the Benchmarks

All benchmark scripts are located at the repository root. They require `piquassoboost` to be built and installed, and the DFE shared library to be on `LD_LIBRARY_PATH` when targeting hardware.

### Permanent Calculator Benchmark

Compares CPU (double, long-double, MPFR) vs. single DFE vs. dual DFE implementations of the Glynn/BB/FG permanent for matrices up to 40×40:

```bash
python3 permanent_benchmark.py
```

### Boson Sampling Simulation Benchmark

Benchmarks the full Clifford & Clifford (2020) boson sampling simulation for a 60-mode interferometer with photon counts up to 40, using the `GeneralizedCliffordsBSimulationStrategy`:

```bash
python3 boson_sampling_benchmark.py
```

### 0-1 Permanent Benchmark

Tests Ryser and Glynn variants for binary matrices and the DFE backend via `ZOnePermanentCalculator`:

```bash
python3 zonepermanent_benchmark.py
```

### Loop Hafnian Benchmark

Compares `thewalrus` reference against `PowerTrace*` CPU implementations (double, long-double, MPFR, recursive):

```bash
python3 loop_hafnian_benchmark.py
```

---

## Performance Highlights

The following results are taken from the published paper (see §3–4 for complete benchmarks).

**Hardware:** Two AMD EPYC 7542 servers (64 threads each), each with two Xilinx Alveo U250 FPGA cards (4 DFEs total). DFE clock: 330 MHz.

| Metric | Result |
|---|---|
| Permanent of a 40×40 unitary (dual DFE) | ~207 s (~337 GOPS) |
| DFE speedup vs. long-double BB/FG CPU | ~27.9× at 40 photons |
| DFE speedup vs. double BB/FG CPU | ~7.4× at 40 photons |
| Numerical accuracy vs. extended precision | Equivalent up to *n* = 40 |
| Boson sampling (40 photons, 60 modes, ideal) | ~80 s/sample (4 DFEs) |
| Boson sampling (40 photons, 60 modes, 30% loss) | ~163 s/sample (4 DFEs) |

For context: computing the permanent of a 45×45 matrix on 98,304 CPU cores (Tianhe-2 supercomputer) takes 24 s via Ryser's formula in double precision. A single server with two DFE engines achieves comparable performance at only ~8.6× slower, without supercomputer resources.

---

## Authors

- **Gregory Morse** — [GregoryMorse](https://github.com/GregoryMorse) — gregory.morse@live.com
- **Tomasz Rybotycki**
- **Ágoston Kaposi**
- **Zoltán Kolarovszki**
- **Uroš Stojčić**
- **Tamás Kozsik**
- **Oskar Mencer**
- **Michał Oszmaniec**
- **Zoltán Zimborás**
- **Péter Rakyta**

---

## Acknowledgments

This research was supported by the Ministry of Culture and Innovation and the National Research, Development and Innovation Office within the **Quantum Information National Laboratory of Hungary** (Grant No. 2022-2.1.1-NL-2022-00004), by the **ÚNKP-22-5** New National Excellence Program, and by the Hungarian Scientific Research Fund (**OTKA**) Grants Nos. K134437 and FK135220. Additional funding was provided by the **QuantERA II** project HQCC-101017733. R.P. acknowledges support from the Hungarian Academy of Sciences through the **Bolyai János Stipendium** (BO/00571/22/11). T.R. and M.O. acknowledge financial support by the Foundation for Polish Science through the **TEAM-NET** project (Contract No. POIR.04.04.00-00-17C1/18-00), and by IRAP **AstroCeNT** (MAB/2018/7) funded by FNP from ERDF. Computational resources were provided by the **Wigner Scientific Computational Laboratory (WSCLAB)**.

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

Portions of the `piquassoboost/` directory are derived from the [piquassoboost](https://github.com/Budapest-Quantum-Computing-Group/piquassoboost) library (Apache 2.0, Budapest Quantum Computing Group).

The original paper is published open access under **Creative Commons Attribution 4.0 (CC BY 4.0)**:
https://creativecommons.org/licenses/by/4.0/

---

## Citation

If you use this code or build on the methods described in the paper, please cite:

```bibtex
@article{Morse_2024,
  title     = {High performance {Boson} sampling simulation via data-flow engines},
  author    = {Gregory Morse and Tomasz Rybotycki and \'{A}goston Kaposi and
               Zolt\'{a}n Kolarovszki and Uro\v{s} Stoj\v{c}i\'{c} and
               Tam\'{a}s Kozsik and Oskar Mencer and Micha{\l} Oszmaniec and
               Zolt\'{a}n Zimbor\'{a}s and P\'{e}ter Rakyta},
  journal   = {New Journal of Physics},
  volume    = {26},
  number    = {3},
  pages     = {033033},
  year      = {2024},
  month     = mar,
  publisher = {IOP Publishing},
  doi       = {10.1088/1367-2630/ad313b},
  url       = {https://doi.org/10.1088/1367-2630/ad313b}
}
```

