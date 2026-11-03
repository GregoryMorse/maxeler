# Plan: verifying the DFE permanent engine (not started)

This is a plan only. Nothing in this document is formalized yet. It describes how
the exact-arithmetic results in this folder could be extended into an end-to-end
statement about what the MaxCompiler DFE design in
`PermanentGlynn/PermanentGlynnDFE` and `PermanentGlynn/PermRepGlynnDFE` computes,
with an explicit error bound.

## 0. Target theorem

For an `n × n` complex input matrix `A` with entries representable in the input
format (`dfeFixOffset(64, -62)`: 64-bit two's complement with 62 fraction bits,
so `|re|, |im| < 2`), and `n ≤ 40` (the engine's limit) and a
row-normalisation assumption like the one the hardware makes (for example
`‖row‖∞ ≤ 2^-⌈log₂ n⌉`, which is what the precision schedule assumes):

```
|DFE(A) - 2^(n-1) · perm A| ≤ ε(n, A)
```

`DFE(A)` is the value the design leaves in the 256-bit accumulator, read as an
exact rational; `ε` is an explicit bound. The cleanest form of `ε` is relative
to `∑_δ ∏_j |colsum_δ j|`, the condition-number-like quantity that every
fixed-point Glynn implementation is controlled by. An absolute bound in terms of
`n` and `‖A‖` then follows from it.

The exact part, `glynnGray_eq` / `glynnRepGray_eq_glynnRep` in this folder, takes
care of `2^(n-1) · perm A = ∑_δ …`. What is left is the error of each arithmetic
stage against that exact sum.

## 1. Model the arithmetic, not the HDL

Put the hardware model in a separate library (`MaxelerPermanent/DFE/`) so the
`Challenge.lean` surface stays small.

1. **Fixed-point numbers.** `structure Fix (w f : ℕ)` holds an integer `m` with
   `-2^(w-1) ≤ m < 2^(w-1)` and denotes `m / 2^f : ℚ`. Model `cast` (truncating
   or rounding, whichever MaxCompiler does), `addExact` (the width grows by one,
   so no error), `mulExact` (the exact product followed by truncation to `bits`),
   and `maxIntegerBits`. Each operation gets a lemma
   `|⟦op x y⟧ - op ⟦x⟧ ⟦y⟧| ≤ 2^-f'` plus a no-overflow side condition.
2. **Complex fixed point.** A pair of `Fix`. State the **3M (Knuth) formula** used
   by `Utility.doMult` exactly:
   `(a+bi)(c+di) = (c(a+b) - b(c+d)) + (c(a+b) + a(d-c)) i`.
   Proving it is `ring`. The work is in the error lemma: three truncated products
   plus the extra integer bits ("2..-2 requires 3 integer bits") give
   `|err_re|, |err_im| ≤ 2 · 2^-bits'`. Do the same for the Ungar/Gauss variant
   if it is ever enabled (`useUngar`).
3. **Rectangular Karatsuba multiplier.** `mulKaratsubaRectangularExact` splits
   the operands into DSP-sized chunks (`getBitChunks128`, 24×17 or 25×18 tiles).
   Prove that the chunked schoolbook/Karatsuba recombination equals the exact
   integer product. This is pure integer arithmetic, provable by `ring` after
   unfolding the chunk decomposition (`Nat`/`Int` `div`/`mod` by `2^k`). If the
   design drops low partial products (the comments say "unclear if an
   approximation algorithm is used"), model the dropped part and bound it. This
   is the one place where the design has to be read carefully against the
   generated hardware.
4. **Floating point to fixed point at the input.** The CPU side
   (`PermanentGlynnCPU`, `PermRepGlynnCPU`) converts `double` to the 64-bit
   format. Model it as rounding to `2^-62` and bound the error by `2^-63` per
   entry. Matrix normalisation (scaling rows by powers of two, undone at the end)
   is exact and is a `perm_smul_row` lemma.

## 2. Stage-by-stage error propagation

Follow the data flow of `PermanentGlynnDFEManager`:

| Stage | Kernel | Exact counterpart | Error source |
|---|---|---|---|
| column-sum init + Gray counter | `InitializeColSumDFEKernel` | `glynnGray` loop state `sums` | none if every column sum fits (additions are exact); prove the width suffices |
| Gray-code update `colsum ± 2·row` | `InitializeColSumDFEKernel` | `glynnGrayStep` | exact (doubling is a shift); the counter equals `grayCode`/`trailingZeros` (already proved) |
| product tree over `n` column sums | `PermanentGlynnDFEKernel` | `∏ j, sums j` | truncation at every tree level; precision grows per level (`totalprec = 64 + errorTol/2`, at least 80 bits) |
| sign + accumulate | `SumUpPermDFEKernel` | `acc + sign * ∏` | the accumulator is wide enough to be exact; prove no overflow over `2^(n-1)` terms |
| 4× stream multiplexing, SLR split, dual-DFE split | managers | splitting a sum into partial sums | exact; it re-associates `∑` |

Two lemmas carry most of this:

* **Product-tree lemma.** For a balanced binary tree of multiplications where
  level `ℓ` truncates to `p_ℓ` fraction bits, and all leaves satisfy `|x_j| ≤ B`,
  `|tree(x) - ∏ x_j| ≤ ∑_ℓ (#nodes_ℓ) · B^(n - 2^ℓ) · 2^-p_ℓ · (1 + small)`.
  Prove it once for an abstract "multiply with error ≤ η_ℓ" operation by
  induction on the tree, then instantiate with the 3M error lemma. The
  precision schedule in `PermanentGlynnDFEKernel.maxj` (`errorTol`, `tol = ⌈log₂ cols⌉`)
  is exactly the choice of `p_ℓ` that makes the right-hand side small.
* **Accumulation lemma.** `∑_k (y_k + e_k) - ∑_k y_k = ∑_k e_k`, and the
  accumulator does not overflow. This is trivial once widths are modelled.

## 3. The repeated-row (boson sampling) variant

`PermRepGlynnDFE` uses the n-ary Gray code. The exact side is
`glynnRepGray_eq_glynnRep`. The hardware side adds:

* multiplicities in the column-sum update `colsum ± 2·row` (still exact);
* powers `colsum^N_l` computed in the product tree by repeating a leaf
  `N_l` times (same product-tree lemma);
* the binomial weight. On the CPU it is an exact `int64`, as in
  `binomStep_eq`; on the DFE, check where it is applied and whether it can
  overflow.

## 4. Linking the model to the MaxJ source

There are three levels of assurance, from cheapest to strongest:

1. **Hand transcription (recommended first).** Transcribe each kernel's
   arithmetic as a Lean function over `Fix`, citing `.maxj` line numbers in
   docstrings. Cross-check it with differential tests: run the Lean model,
   compiled with `#eval` and arbitrary-precision rationals, against the MaxCompiler
   simulator (`make RUNRULE=Simulation`) on random matrices; the outputs must
   agree bit for bit.
2. **Extracted dataflow graph.** MaxCompiler can dump the kernel graph. A small
   translator from that graph to Lean terms would remove the transcription step.
3. **Netlist level.** Out of scope.

## 5. Suggested milestones

1. `Fix` arithmetic, cast/add/mul error lemmas.
2. The 3M identity (`ring`) and its error lemma.
3. The abstract product-tree error lemma.
4. A Lean model of `PermanentGlynnDFEKernel` at the parameters used for `n ≤ 40`,
   and the end-to-end bound for the non-repeated engine.
5. The Karatsuba rectangular multiplier: an exactness proof, or a bound for
   dropped partial products.
6. The repeated-row engine.
7. Optional: a numeric instantiation of `ε(40, ·)` for unitary submatrices,
   compared against the `long double` / MPFR reference results in
   `permanent_benchmark.py`.

## 6. Practical notes

* Keep the hardware model and its error theorems out of `Challenge.lean` until
  they are stable. A Palomar statement should be the final inequality, with the
  fixed-point model as audited definitions.
* Exact rationals (`ℚ`) are the right carrier: every fixed-point value is a
  dyadic rational, so no real analysis is needed except to state relative bounds.
* `norm_num`/`decide` can discharge the per-level width side conditions once
  `n` is fixed. For the general `n ≤ 40` statement, prove them uniformly or
  case-split over the 40 sizes.
