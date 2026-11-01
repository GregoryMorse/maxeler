# Lean formalization: exact permanent algorithms

This folder is a Lean 4 / Mathlib project. It proves, over any commutative ring,
that the exact permanent algorithms behind the boson sampling engines in this
repository compute the permanent. The engines are described in
[Morse et al., *New J. Phys.* 26, 033033 (2024)](https://doi.org/10.1088/1367-2630/ad313b).
The textbook determinant is included as well.

| Algorithm | Lean definition | Theorem (`Challenge.lean`) |
|---|---|---|
| Naive permanent `∑_σ ∏_i a_{i,σ(i)}` | `permNaive` | `permNaive_correct` |
| Naive (Leibniz) determinant | `detNaive` | `detNaive_correct` |
| Ryser | `ryser` | `ryser_correct` |
| BB/FG (Glynn), `δ₀ = +1` | `glynn` | `glynn_correct` (`= 2ⁿ · perm`) |
| Ryser, binary Gray code (Nijenhuis–Wilf) | `ryserGray` | `ryserGray_correct` |
| BB/FG, binary Gray code | `glynnGray` | `glynnGray_correct` |
| Ryser with repeated rows/columns | `ryserRep` | `ryserRep_correct` |
| BB/FG with repeated rows/columns | `glynnRep` | `glynnRep_correct` |
| Ryser with repetitions, n-ary Gray code | `ryserRepGray` | `ryserRepGray_correct` |
| BB/FG with repetitions, n-ary Gray code | `glynnRepGray` | `glynnRepGray_correct` |

The reference is Mathlib's `Matrix.permanent` / `Matrix.det`. Repeated rows and
columns are stated for `A.submatrix r c` with arbitrary maps `r, c : Fin n → Fin _`:
row `k` occurs `mult r k` times and column `l` occurs `mult c l` times. This is the
matrix whose permanent the generalised Clifford & Clifford sampler needs.
`repeatMatrix` (in `Repeated.lean`) gives the canonical block-repeated matrix.

The Gray-code algorithms are written as loops (`runSteps`) that mirror the C++
reference implementation:

* **Binary.** Step `k` uses `j = trailingZeros k` and the bit `j` of
  `grayCode k = k ⊕ (k >> 1)`. One column is added to or removed from the row sums
  (Ryser), or one row sign flips and `±2·row` is applied to the column sums
  (BB/FG). The running sign flips each step.
* **n-ary.** `naryGray` is the reflected mixed-radix Gray code. At each step one
  digit `i = naryGrayIdx` moves by ±1, the sums are updated by `±row/column i`,
  and the binomial weight is updated by exact integer division (`binomStep`), as
  in `piquassoboost/sampling/source/BBFGPermanentCalculatorRepeated.hpp`.

The fixed-point FPGA implementation is **not** formalized.
[`FPGA_VERIFICATION_PLAN.md`](FPGA_VERIFICATION_PLAN.md) sketches how an
end-to-end error bound could be proved. It would cover the float-to-fixed
conversion, the 3M complex multiplier, the Karatsuba DSP tiling, and the
product tree whose precision grows level by level.

## Layout

```
Challenge.lean            statement surface (definitions + theorems with `sorry`)
Solution.lean             the same theorems, proved from the library
MaxelerPermanent/
  Defs.lean               all definitions (copied verbatim into Challenge.lean)
  Basic.lean              naive formulas, Ryser, BB/FG
  Repeated.lean           multiplicity formulas, canonical repeated matrix
  GrayCode.lean           binary and n-ary Gray codes: bijectivity, adjacency
  GrayAlgorithms.lean     loop invariants for the four Gray-code algorithms
comparator.json           Palomar / `lake comparator` configuration
formalization.yaml        Palomar metadata
scripts/                  Palomar pre-flight scripts (+ sync-challenge.py)
docbuild/                 doc-gen4 project
```

## Building

```bash
cd formal
```

```bash
lake exe cache get
```

```bash
lake build
```

## Palomar pre-flight

These are the checks the Palomar template runs. CI
(`.github/workflows/formal.yml`) runs all of them.

```bash
python3 scripts/check-lean-sources.py
```

```bash
python3 scripts/sync-challenge.py --check
```

```bash
ruby scripts/validate-formalization.rb
```

```bash
./scripts/verify-comparator.sh
```

`verify-comparator.sh` needs `bwrap` (bubblewrap). Lean v4.35.0 or later is
required, because that is the first release that ships `lake comparator`.

Palomar registers a GitHub repository at a commit, and it expects the Lake
project at the repository root. To submit, either mirror this folder into its own
repository, or check that Palomar accepts a subdirectory project. Then open the
submission form at https://submit.palomar-registry.org/ with the full commit SHA.
Read the [Palomar submission policy](https://github.com/PalomarRegistry/PalomarPolicy/blob/main/CONTRIBUTING.md)
first, and update the `review` section of `formalization.yaml` after a human audit.
