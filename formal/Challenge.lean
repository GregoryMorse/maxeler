module

public import Mathlib.LinearAlgebra.Matrix.Permanent
public import Mathlib.LinearAlgebra.Matrix.Determinant.Basic

/-!
# Advertised statements

Correctness and exact operation counts of exact permanent algorithms used for boson sampling simulation in
*High performance Boson sampling simulation via data-flow engines*
(Morse et al., New J. Phys. 26, 033033, 2024), and of the textbook determinant.

The definitions below are copied verbatim from `MaxelerPermanent/Defs.lean`
(regenerate with `python3 scripts/sync-challenge.py`). Each theorem compares an
algorithm with Mathlib's `Matrix.permanent` (`∑_σ ∏_i M (σ i) i`) or
`Matrix.det`, over an arbitrary commutative ring.

Repeated rows/columns are expressed through `A.submatrix r c` for arbitrary
`r : Fin n → Fin p`, `c : Fin n → Fin q`: row `k` of `A` occurs `mult r k`
times and column `l` occurs `mult c l` times.
-/

universe u

@[expose] public section

open Finset

-- BEGIN DEFINITIONS (generated from MaxelerPermanent/Defs.lean)

namespace MaxelerPermanent

variable {R : Type*} [CommRing R]

/-! ## Naive definitions -/

/-- The permanent as the textbook sum over permutations,
`perm A = ∑_σ ∏_i A i (σ i)`. -/
def permNaive {n : ℕ} (A : Matrix (Fin n) (Fin n) R) : R :=
  ∑ σ : Equiv.Perm (Fin n), ∏ i, A i (σ i)

/-- The determinant as the textbook (Leibniz) sum over permutations,
`det A = ∑_σ sign σ ∏_i A i (σ i)`. -/
def detNaive {n : ℕ} (A : Matrix (Fin n) (Fin n) R) : R :=
  ∑ σ : Equiv.Perm (Fin n), ((Equiv.Perm.sign σ : ℤ) : R) * ∏ i, A i (σ i)

/-! ## Ryser and Glynn (BB/FG) -/

/-- Ryser's formula: `∑_{S ⊆ cols} (-1)^(n - |S|) ∏_i ∑_{j ∈ S} A i j`. -/
def ryser {n : ℕ} (A : Matrix (Fin n) (Fin n) R) : R :=
  ∑ S : Finset (Fin n), (-1) ^ (n - #S) * ∏ i, ∑ j ∈ S, A i j

/-- The sign `δ_i` of row `i` when `S` is the set of rows with `δ_i = -1`. -/
def flipSign {ι : Type*} [DecidableEq ι] (S : Finset ι) (i : ι) : R :=
  if i ∈ S then -1 else 1

/-- The BB/FG (Glynn) sum for an `(n+1) × (n+1)` matrix with `δ₀ = +1` fixed:
`∑_{δ ∈ {±1}ⁿ} (∏ δ) ∏_j (A 0 j + ∑_i δ_i A (i+1) j)`, where `S` is the set of
rows `i + 1` with `δ = -1`. It equals `2ⁿ · perm A`. -/
def glynn {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) : R :=
  ∑ S : Finset (Fin n), (-1) ^ #S *
    ∏ j, (A 0 j + ∑ i : Fin n, flipSign S i * A i.succ j)

/-! ## Binary Gray code -/

/-- The binary reflected Gray code, `k ⊕ (k >> 1)`. -/
def grayCode (k : ℕ) : ℕ := k ^^^ (k >>> 1)

/-- The number of trailing zero bits of `k` (with `trailingZeros 0 = 0`); this is
the bit in which `grayCode (k - 1)` and `grayCode k` differ. -/
def trailingZeros (k : ℕ) : ℕ :=
  if h : k = 0 then 0 else if k % 2 = 1 then 0 else trailingZeros (k / 2) + 1
termination_by k
decreasing_by omega

/-- `natIdx f i = f i` for `i < m` and `0` otherwise: indexing a `Fin m`-vector by a
natural number, as array code does. -/
def natIdx {m : ℕ} {α : Type*} [Zero α] (f : Fin m → α) (i : ℕ) : α :=
  if h : i < m then f ⟨i, h⟩ else 0

/-- Run a loop `for k in 1..t: s ← step k s` from `init`. -/
def runSteps {S : Type*} (step : ℕ → S → S) (init : S) : ℕ → S
  | 0 => init
  | t + 1 => step (t + 1) (runSteps step init t)

/-- Loop state of the Gray-code permanent algorithms: running row/column sums,
the current sign, and the accumulated sum. -/
structure GrayState (ι : Type*) (R : Type*) where
  /-- running row sums (Ryser) or column sums (Glynn) -/
  sums : ι → R
  /-- sign of the current term -/
  sign : R
  /-- accumulated sum -/
  acc : R

/-- One step of Ryser in Gray-code order: at step `k`, column `j = tz k` enters
(if bit `j` of `grayCode k` is set) or leaves the subset. -/
def ryserGrayStep {n : ℕ} (A : Matrix (Fin n) (Fin n) R) (k : ℕ)
    (s : GrayState (Fin n) R) : GrayState (Fin n) R :=
  let j := trailingZeros k
  let sums := if (grayCode k).testBit j then s.sums + (fun i => natIdx (A i) j) else s.sums - (fun i => natIdx (A i) j)
  let sign := -s.sign
  ⟨sums, sign, s.acc + sign * ∏ i, sums i⟩

/-- Ryser's formula evaluated in binary Gray-code order (Nijenhuis–Wilf):
`perm A = (-1)ⁿ ∑_{k < 2ⁿ} (-1)^|S_k| ∏_i r_i`, where `S_k` is the column set
encoded by `grayCode k` and the row sums `r_i` are updated by one column per step.
The `k = 0` term (`∏_i 0`) only contributes when `n = 0`. -/
def ryserGray {n : ℕ} (A : Matrix (Fin n) (Fin n) R) : R :=
  (-1) ^ n * (runSteps (ryserGrayStep A) ⟨0, 1, ∏ _i : Fin n, (0 : R)⟩ (2 ^ n - 1)).acc

/-- One step of BB/FG in Gray-code order: at step `k`, the sign of row `tz k + 1`
flips; its doubled row is subtracted from (sign becomes `-1`, i.e. bit set) or added
to the column sums. -/
def glynnGrayStep {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) (k : ℕ)
    (s : GrayState (Fin (n + 1)) R) : GrayState (Fin (n + 1)) R :=
  let j := trailingZeros k
  let sums := if (grayCode k).testBit j then s.sums - 2 • natIdx A (j + 1)
    else s.sums + 2 • natIdx A (j + 1)
  let sign := -s.sign
  ⟨sums, sign, s.acc + sign * ∏ i, sums i⟩

/-- BB/FG evaluated in binary Gray-code order over the `2ⁿ` sign vectors with
`δ₀ = +1`; equals `2ⁿ · perm A`. The column sums start at `∑_i A i j`. -/
def glynnGray {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) : R :=
  let init : Fin (n + 1) → R := fun j => ∑ i, A i j
  (runSteps (glynnGrayStep A) ⟨init, 1, ∏ j, init j⟩ (2 ^ n - 1)).acc

/-! ## Repeated rows and columns -/

/-- Multiplicities of a selection `r : Fin n → Fin p`: `mult r k = #{i | r i = k}`.
`A.submatrix r c` repeats row `k` of `A` exactly `mult r k` times and column `l`
exactly `mult c l` times. -/
def mult {n p : ℕ} (r : Fin n → Fin p) (k : Fin p) : ℕ :=
  #{i | r i = k}

/-- Ryser's formula with multiplicities: with column multiplicities `N` and row
multiplicities `M`,
`∑_{0 ≤ s ≤ N} (-1)^(|N| - |s|) (∏_l C(N_l, s_l)) ∏_k (∑_l s_l A k l)^(M_k)`. -/
def ryserRep {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (M : Fin p → ℕ) (N : Fin q → ℕ) : R :=
  ∑ s ∈ Fintype.piFinset (fun l => range (N l + 1)),
    (-1) ^ (∑ l, N l - ∑ l, s l) * ((∏ l, (N l).choose (s l) : ℕ) : R) *
      ∏ k, (∑ l, (s l : R) * A k l) ^ M k

/-- BB/FG with multiplicities: row `k0` of `A` is used once with fixed sign `+1`;
the rows have further multiplicities `M`, the columns multiplicities `N`; `d_k` is
the number of copies of row `k` carrying sign `-1`:
`∑_{0 ≤ d ≤ M} (-1)^|d| (∏_k C(M_k, d_k)) ∏_l (A k0 l + ∑_k (M_k - 2 d_k) A k l)^(N_l)`.
It equals `2^|M|` times the permanent of the repeated matrix. -/
def glynnRep {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (k0 : Fin p)
    (M : Fin p → ℕ) (N : Fin q → ℕ) : R :=
  ∑ d ∈ Fintype.piFinset (fun k => range (M k + 1)),
    (-1) ^ (∑ k, d k) * ((∏ k, (M k).choose (d k) : ℕ) : R) *
      ∏ l, (A k0 l + ∑ k, ((M k : R) - 2 * (d k : R)) * A k l) ^ N l

/-! ## n-ary (mixed-radix) Gray code -/

/-- The reflected mixed-radix Gray code with radices `rad`: digit `0` runs fastest
and reverses direction whenever the higher digits advance.
`naryGray rad t` enumerates `∏_k [0, rad k)` without repetition for `t < ∏ rad`. -/
def naryGray : {m : ℕ} → (Fin m → ℕ) → ℕ → Fin m → ℕ
  | 0, _, _ => fun i => i.elim0
  | _ + 1, rad, t =>
    Fin.cons (if t / rad 0 % 2 = 0 then t % rad 0 else rad 0 - 1 - t % rad 0)
      (naryGray (Fin.tail rad) (t / rad 0))

/-- The digit in which `naryGray rad t` and `naryGray rad (t + 1)` differ. -/
def naryGrayIdx : {m : ℕ} → (Fin m → ℕ) → ℕ → ℕ
  | 0, _, _ => 0
  | _ + 1, rad, t =>
    if t % rad 0 + 1 < rad 0 then 0 else naryGrayIdx (Fin.tail rad) (t / rad 0) + 1

/-- Loop state of the n-ary Gray-code algorithms: the running sums, the sign, the
binomial weight (a natural number, updated by exact division), and the accumulator. -/
structure RepState (ι : Type*) (R : Type*) where
  /-- running row sums (Ryser) or column sums (Glynn) -/
  sums : ι → R
  /-- sign of the current term -/
  sign : R
  /-- current product of binomial coefficients -/
  binom : ℕ
  /-- accumulated sum -/
  acc : R

/-- Update a binomial coefficient `C(m, u)` to `C(m, v)` for `v = u ± 1`, by exact
division as in the C++ implementation. -/
def binomStep (b m u v : ℕ) : ℕ :=
  if v < u then b * u / (m - v) else b * (m - u) / v

/-- One step `t - 1 → t` of Ryser with multiplicities in n-ary Gray-code order:
digit `i` of the column occupation vector `s` changes from `u` to `v = u ± 1`;
column `i` is added to (`u < v`) or subtracted from the row sums. -/
def ryserRepGrayStep {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (M : Fin p → ℕ)
    (N : Fin q → ℕ) (t : ℕ) (st : RepState (Fin p) R) : RepState (Fin p) R :=
  let rad : Fin q → ℕ := fun l => N l + 1
  let i := naryGrayIdx rad (t - 1)
  let u := natIdx (naryGray rad (t - 1)) i
  let v := natIdx (naryGray rad t) i
  let sums := if u < v then st.sums + (fun k => natIdx (A k) i)
    else st.sums - (fun k => natIdx (A k) i)
  let sign := -st.sign
  let binom := binomStep st.binom (natIdx N i) u v
  ⟨sums, sign, binom, st.acc + sign * (binom : R) * ∏ k, sums k ^ M k⟩

/-- Ryser with multiplicities evaluated in n-ary Gray-code order over the column
occupation vectors `0 ≤ s ≤ N`, with incrementally updated row sums
`∑_l s_l A k l` and binomial weight `∏_l C(N_l, s_l)`. Equals `ryserRep A M N`. -/
def ryserRepGray {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (M : Fin p → ℕ)
    (N : Fin q → ℕ) : R :=
  (-1) ^ (∑ l, N l) * (runSteps (ryserRepGrayStep A M N)
    ⟨0, 1, 1, ∏ k, (0 : R) ^ M k⟩ (∏ l, (N l + 1) - 1)).acc

/-- One step `t - 1 → t` of BB/FG with multiplicities in n-ary Gray-code order:
the number `d_i` of `-1` signs among the copies of row `i` changes from `u` to
`v = u ± 1`; twice row `i` is subtracted from (`u < v`) or added to the column sums.
This is the loop body of `BBFGPermanentCalculatorRepeated.hpp`. -/
def glynnRepGrayStep {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (M : Fin p → ℕ)
    (N : Fin q → ℕ) (t : ℕ) (st : RepState (Fin q) R) : RepState (Fin q) R :=
  let rad : Fin p → ℕ := fun k => M k + 1
  let i := naryGrayIdx rad (t - 1)
  let u := natIdx (naryGray rad (t - 1)) i
  let v := natIdx (naryGray rad t) i
  let sums := if u < v then st.sums - 2 • natIdx A i else st.sums + 2 • natIdx A i
  let sign := -st.sign
  let binom := binomStep st.binom (natIdx M i) u v
  ⟨sums, sign, binom, st.acc + sign * (binom : R) * ∏ l, sums l ^ N l⟩

/-- BB/FG with multiplicities evaluated in n-ary Gray-code order over `0 ≤ d ≤ M`,
starting from the column sums `A k0 l + ∑_k M_k A k l` (all signs `+1`).
Equals `glynnRep A k0 M N`. -/
def glynnRepGray {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (k0 : Fin p)
    (M : Fin p → ℕ) (N : Fin q → ℕ) : R :=
  let init : Fin q → R := fun l => A k0 l + ∑ k, (M k : R) * A k l
  (runSteps (glynnRepGrayStep A M N) ⟨init, 1, 1, ∏ l, init l ^ N l⟩
    (∏ k, (M k + 1) - 1)).acc

/-! ## BB/FG in the form of `permanent_benchmark.py` (Wikipedia) -/

/-- Loop state of `permanent_glynn_gray_exact` in `permanent_benchmark.py`: the sign
vector `δ` (`true` = `+1`), the column sums and the accumulated total. -/
structure WikiState (n : ℕ) (R : Type*) where
  /-- `delta i = true` iff `δ_i = +1`, for the `n` rows whose sign varies -/
  delta : Fin n → Bool
  /-- running column sums `∑_i δ_i A i j` -/
  sums : Fin (n + 1) → R
  /-- accumulated total -/
  acc : R

/-- Toggle entry `i` (a natural-number index) of a Boolean vector, like
`gcode[idx] = 1 - gcode[idx]`. -/
def toggleAt {n : ℕ} (δ : Fin n → Bool) (i : ℕ) : Fin n → Bool :=
  fun k => if k.val = i then !δ k else δ k

/-- One step `i` of `permanent_glynn_gray_exact`: toggle `δ` at `idx = tz i`, add
(if `δ_idx` became `+1`) or subtract twice row `idx` from the column sums, and add
(`i` even) or subtract (`i` odd) the product of the column sums. -/
def glynnWikiStep {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) (i : ℕ)
    (s : WikiState n R) : WikiState n R :=
  let idx := trailingZeros i
  let delta := toggleAt s.delta idx
  let sums := if (if h : idx < n then delta ⟨idx, h⟩ else false) then s.sums + 2 • natIdx A idx
    else s.sums - 2 • natIdx A idx
  let p := ∏ j, sums j
  ⟨delta, sums, if i % 2 = 1 then s.acc - p else s.acc + p⟩

/-- BB/FG in Gray-code order exactly as in `permanent_glynn_gray_exact`
(`permanent_benchmark.py`; Wikipedia's Gray-code form of Glynn's formula): the last
row has fixed sign `+1`, the term signs alternate with the step counter, and
`δ` is toggled in place. Equals `2ⁿ · perm A` (before the division by `2ⁿ`). -/
def glynnWiki {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) : R :=
  let init : Fin (n + 1) → R := fun j => ∑ i, A i j
  (runSteps (glynnWikiStep A) ⟨fun _ => true, init, ∏ j, init j⟩ (2 ^ n - 1)).acc

/-! ## Shifted Ryser (Nijenhuis–Wilf) and normalized BB/FG -/

/-- Ryser's formula with the row sums shifted by half the row total
(Nijenhuis–Wilf), scaled by `2` to avoid division: for an `(n+1) × (n+1)` matrix,
`∑_{S ⊆ first n cols} (-1)^(n - |S|) ∏_i (2 a_{i,last} - ∑_j a_{ij} + 2 ∑_{j ∈ S} a_{ij})`,
which equals `2ⁿ · perm A`. Only `2ⁿ` subsets of the first `n` columns are
needed instead of `2ⁿ⁺¹`. -/
def ryserShift {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) : R :=
  ∑ S : Finset (Fin n), (-1) ^ (n - #S) *
    ∏ i, (2 * A i (Fin.last n) - ∑ j, A i j + 2 * ∑ j ∈ S, A i j.castSucc)

/-- One step of shifted Ryser in Gray-code order: column `j = tz k` enters (bit
set; `+2·col`) or leaves (`-2·col`) the subset. -/
def ryserShiftGrayStep {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) (k : ℕ)
    (s : GrayState (Fin (n + 1)) R) : GrayState (Fin (n + 1)) R :=
  let j := trailingZeros k
  let sums := if (grayCode k).testBit j then s.sums + 2 • (fun i => natIdx (A i) j)
    else s.sums - 2 • (fun i => natIdx (A i) j)
  let sign := -s.sign
  ⟨sums, sign, s.acc + sign * ∏ i, sums i⟩

/-- The traditional Nijenhuis–Wilf algorithm: shifted Ryser in binary Gray-code
order, with row sums starting at `2 a_{i,last} - ∑_j a_{ij}` and sign `(-1)ⁿ`.
Equals `2ⁿ · perm A`. -/
def ryserShiftGray {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) : R :=
  let init : Fin (n + 1) → R := fun i => 2 * A i (Fin.last n) - ∑ j, A i j
  (runSteps (ryserShiftGrayStep A) ⟨init, (-1) ^ n, (-1) ^ n * ∏ i, init i⟩ (2 ^ n - 1)).acc

section Field

variable {K : Type*} [Field K]

/-- The Nijenhuis–Wilf formula as published, over a field:
`perm A = (-1)ⁿ · 2 · ∑_{S ⊆ first n cols} (-1)^|S| ∏_i (x_i + ∑_{j ∈ S} a_{ij})`
with `x_i = a_{i,last} - ½ ∑_j a_{ij}` (requires `2 ≠ 0`). -/
def ryserNW {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K) : K :=
  (-1) ^ n * 2 * ∑ S : Finset (Fin n), (-1) ^ #S *
    ∏ i, (A i (Fin.last n) - (∑ j, A i j) / 2 + ∑ j ∈ S, A i j.castSucc)

/-- Normalized BB/FG, as in `permanent_glynn_gray_fixpt` (`permanent_benchmark.py`):
divide row `i` by a nonzero scale `r i`, run the Gray-code BB/FG loop `glynnWiki`,
and multiply the result by `∏ r i`. Equals `2ⁿ · perm A` when every `r i ≠ 0`. -/
def glynnNorm {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K) (r : Fin (n + 1) → K) : K :=
  (∏ i, r i) * glynnWiki (Matrix.of fun i j => A i j / r i)

end Field

/-! ## Cost model

`Costed α` pairs a value with the number of arithmetic operations in the ring
used to compute it. The cost-instrumented programs `…C` below re-express each
algorithm with explicit operation counts:

* one addition, subtraction, negation or multiplication of ring elements costs `1`
  (multiplying a matrix entry by a small integer coefficient or by `±1` also
  counts as one multiplication; adding two length-`k` vectors costs `k`);
* `bigSum s f` costs the costs of the terms plus `#s` additions (accumulating from
  `0`), `bigProd s f` the costs of the factors plus `#s` multiplications;
* `x ^ k` costs `k` multiplications (repeated multiplication, as in the C++ code);
* integer bookkeeping (Gray-code counter, trailing zeros, binomial coefficient
  updates, loop indices) and reading matrix entries are free.

The theorems in `Challenge.lean` state `prog A = ⟨algorithm A, exact count⟩`: the
instrumented program computes exactly the algorithm's value with that many ring
operations. -/

/-- A value together with the number of ring operations used to compute it. -/
@[ext]
structure Costed (α : Type u) where
  /-- the computed value -/
  val : α
  /-- the number of ring operations used -/
  cost : ℕ

instance : Monad Costed where
  pure a := ⟨a, 0⟩
  bind x f := ⟨(f x.val).val, x.cost + (f x.val).cost⟩

namespace Costed

/-- A value computed with `k` operations. -/
def tick {α : Type u} (a : α) (k : ℕ) : Costed α := ⟨a, k⟩

/-- `∑ i ∈ s, f i`: the costs of the terms plus `#s` additions. -/
def bigSum {ι : Type*} {α : Type u} [AddCommMonoid α] (s : Finset ι) (f : ι → Costed α) :
    Costed α :=
  ⟨∑ i ∈ s, (f i).val, (∑ i ∈ s, (f i).cost) + #s⟩

/-- `∏ i ∈ s, f i`: the costs of the factors plus `#s` multiplications. -/
def bigProd {ι : Type*} {α : Type u} [CommMonoid α] (s : Finset ι) (f : ι → Costed α) :
    Costed α :=
  ⟨∏ i ∈ s, (f i).val, (∑ i ∈ s, (f i).cost) + #s⟩

/-- Build a vector entrywise: the costs of the entries. -/
def vec {ι : Type*} [Fintype ι] {α : Type u} (f : ι → Costed α) : Costed (ι → α) :=
  ⟨fun i => (f i).val, ∑ i, (f i).cost⟩

/-- `for k in 1..T: s ← step k s`. -/
def loop {σ : Type u} (step : ℕ → σ → Costed σ) (init : σ) : ℕ → Costed σ
  | 0 => pure init
  | t + 1 => loop step init t >>= step (t + 1)

end Costed

open Costed

/-- Instrumented naive permanent. -/
def permNaiveC {n : ℕ} (A : Matrix (Fin n) (Fin n) R) : Costed R :=
  bigSum univ fun σ : Equiv.Perm (Fin n) => bigProd univ fun i => pure (A i (σ i))

/-- Instrumented Leibniz determinant. -/
def detNaiveC {n : ℕ} (A : Matrix (Fin n) (Fin n) R) : Costed R :=
  bigSum univ fun σ : Equiv.Perm (Fin n) => do
    let p ← bigProd univ fun i => pure (A i (σ i))
    tick (((Equiv.Perm.sign σ : ℤ) : R) * p) 1

/-- Instrumented Ryser formula (every subset's row sums computed from scratch). -/
def ryserC {n : ℕ} (A : Matrix (Fin n) (Fin n) R) : Costed R :=
  bigSum univ fun S : Finset (Fin n) => do
    let p ← bigProd univ fun i => bigSum S fun j => pure (A i j)
    tick ((-1) ^ (n - #S) * p) 1

/-- Instrumented BB/FG formula (every sign vector's column sums from scratch). -/
def glynnC {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) : Costed R :=
  bigSum univ fun S : Finset (Fin n) => do
    let p ← bigProd univ fun j => do
      let c ← bigSum univ fun i : Fin n => tick (flipSign S i * A i.succ j) 1
      tick (A 0 j + c) 1
    tick ((-1) ^ #S * p) 1

/-- Instrumented loop body of `ryserGray`. -/
def ryserGrayStepC {n : ℕ} (A : Matrix (Fin n) (Fin n) R) (k : ℕ) (s : GrayState (Fin n) R) :
    Costed (GrayState (Fin n) R) := do
  let j := trailingZeros k
  let sums ← if (grayCode k).testBit j then tick (s.sums + fun i => natIdx (A i) j) n
    else tick (s.sums - fun i => natIdx (A i) j) n
  let sign ← tick (-s.sign) 1
  let p ← bigProd univ fun i => pure (sums i)
  let t ← tick (sign * p) 1
  tick ⟨sums, sign, s.acc + t⟩ 1

/-- Instrumented `ryserGray`. -/
def ryserGrayC {n : ℕ} (A : Matrix (Fin n) (Fin n) R) : Costed R := do
  let p0 ← bigProd univ fun _ : Fin n => pure (0 : R)
  let st ← loop (ryserGrayStepC A) ⟨0, 1, p0⟩ (2 ^ n - 1)
  tick ((-1) ^ n * st.acc) 1

/-- Instrumented loop body of `glynnGray`. -/
def glynnGrayStepC {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) (k : ℕ)
    (s : GrayState (Fin (n + 1)) R) : Costed (GrayState (Fin (n + 1)) R) := do
  let j := trailingZeros k
  let sums ← if (grayCode k).testBit j then tick (s.sums - 2 • natIdx A (j + 1)) (2 * (n + 1))
    else tick (s.sums + 2 • natIdx A (j + 1)) (2 * (n + 1))
  let sign ← tick (-s.sign) 1
  let p ← bigProd univ fun i => pure (sums i)
  let t ← tick (sign * p) 1
  tick ⟨sums, sign, s.acc + t⟩ 1

/-- Instrumented `glynnGray`. -/
def glynnGrayC {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) : Costed R := do
  let init ← vec fun j => bigSum univ fun i => pure (A i j)
  let p0 ← bigProd univ fun j => pure (init j)
  let st ← loop (glynnGrayStepC A) ⟨init, 1, p0⟩ (2 ^ n - 1)
  pure st.acc

/-- Instrumented loop body of `glynnWiki`. -/
def glynnWikiStepC {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) (i : ℕ)
    (s : WikiState n R) : Costed (WikiState n R) := do
  let idx := trailingZeros i
  let delta := toggleAt s.delta idx
  let sums ← if (if h : idx < n then delta ⟨idx, h⟩ else false) then tick (s.sums + 2 • natIdx A idx) (2 * (n + 1))
    else tick (s.sums - 2 • natIdx A idx) (2 * (n + 1))
  let p ← bigProd univ fun j => pure (sums j)
  let acc ← if i % 2 = 1 then tick (s.acc - p) 1 else tick (s.acc + p) 1
  pure ⟨delta, sums, acc⟩

/-- Instrumented `glynnWiki`. -/
def glynnWikiC {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) : Costed R := do
  let init ← vec fun j => bigSum univ fun i => pure (A i j)
  let p0 ← bigProd univ fun j => pure (init j)
  let st ← loop (glynnWikiStepC A) ⟨fun _ => true, init, p0⟩ (2 ^ n - 1)
  pure st.acc

/-- Instrumented Ryser formula with multiplicities. -/
def ryserRepC {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (M : Fin p → ℕ) (N : Fin q → ℕ) :
    Costed R :=
  bigSum (Fintype.piFinset fun l => range (N l + 1)) fun s => do
    let pr ← bigProd univ fun k => do
      let r ← bigSum univ fun l => tick ((s l : R) * A k l) 1
      tick (r ^ M k) (M k)
    tick ((-1) ^ (∑ l, N l - ∑ l, s l) * ((∏ l, (N l).choose (s l) : ℕ) : R) * pr) 2

/-- Instrumented BB/FG formula with multiplicities. -/
def glynnRepC {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (k0 : Fin p) (M : Fin p → ℕ)
    (N : Fin q → ℕ) : Costed R :=
  bigSum (Fintype.piFinset fun k => range (M k + 1)) fun d => do
    let pr ← bigProd univ fun l => do
      let c ← bigSum univ fun k => tick (((M k : R) - 2 * (d k : R)) * A k l) 1
      let c' ← tick (A k0 l + c) 1
      tick (c' ^ N l) (N l)
    tick ((-1) ^ (∑ k, d k) * ((∏ k, (M k).choose (d k) : ℕ) : R) * pr) 2

/-- Instrumented loop body of `ryserRepGray`. -/
def ryserRepGrayStepC {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (M : Fin p → ℕ)
    (N : Fin q → ℕ) (t : ℕ) (st : RepState (Fin p) R) : Costed (RepState (Fin p) R) := do
  let rad : Fin q → ℕ := fun l => N l + 1
  let i := naryGrayIdx rad (t - 1)
  let u := natIdx (naryGray rad (t - 1)) i
  let v := natIdx (naryGray rad t) i
  let sums ← if u < v then tick (st.sums + fun k => natIdx (A k) i) p
    else tick (st.sums - fun k => natIdx (A k) i) p
  let sign ← tick (-st.sign) 1
  let binom := binomStep st.binom (natIdx N i) u v
  let pr ← bigProd univ fun k => tick (sums k ^ M k) (M k)
  let term ← tick (sign * (binom : R) * pr) 2
  tick ⟨sums, sign, binom, st.acc + term⟩ 1

/-- Instrumented `ryserRepGray`. -/
def ryserRepGrayC {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (M : Fin p → ℕ)
    (N : Fin q → ℕ) : Costed R := do
  let p0 ← bigProd univ fun k => tick ((0 : R) ^ M k) (M k)
  let st ← loop (ryserRepGrayStepC A M N) ⟨0, 1, 1, p0⟩ (∏ l, (N l + 1) - 1)
  tick ((-1) ^ (∑ l, N l) * st.acc) 1

/-- Instrumented loop body of `glynnRepGray`. -/
def glynnRepGrayStepC {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (M : Fin p → ℕ)
    (N : Fin q → ℕ) (t : ℕ) (st : RepState (Fin q) R) : Costed (RepState (Fin q) R) := do
  let rad : Fin p → ℕ := fun k => M k + 1
  let i := naryGrayIdx rad (t - 1)
  let u := natIdx (naryGray rad (t - 1)) i
  let v := natIdx (naryGray rad t) i
  let sums ← if u < v then tick (st.sums - 2 • natIdx A i) (2 * q)
    else tick (st.sums + 2 • natIdx A i) (2 * q)
  let sign ← tick (-st.sign) 1
  let binom := binomStep st.binom (natIdx M i) u v
  let pr ← bigProd univ fun l => tick (sums l ^ N l) (N l)
  let term ← tick (sign * (binom : R) * pr) 2
  tick ⟨sums, sign, binom, st.acc + term⟩ 1

/-- Instrumented `glynnRepGray`. -/
def glynnRepGrayC {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (k0 : Fin p)
    (M : Fin p → ℕ) (N : Fin q → ℕ) : Costed R := do
  let init ← vec fun l => do
    let c ← bigSum univ fun k => tick ((M k : R) * A k l) 1
    tick (A k0 l + c) 1
  let p0 ← bigProd univ fun l => tick (init l ^ N l) (N l)
  let st ← loop (glynnRepGrayStepC A M N) ⟨init, 1, 1, p0⟩ (∏ k, (M k + 1) - 1)
  pure st.acc

/-- Instrumented shifted Ryser formula (every subset's sums from scratch). -/
def ryserShiftC {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) : Costed R :=
  bigSum univ fun S : Finset (Fin n) => do
    let p ← bigProd univ fun i => do
      let rs ← bigSum univ fun j => pure (A i j)
      let c ← bigSum S fun j => pure (A i j.castSucc)
      tick (2 * A i (Fin.last n) - rs + 2 * c) 4
    tick ((-1) ^ (n - #S) * p) 1

/-- Instrumented loop body of `ryserShiftGray`. -/
def ryserShiftGrayStepC {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) (k : ℕ)
    (s : GrayState (Fin (n + 1)) R) : Costed (GrayState (Fin (n + 1)) R) := do
  let j := trailingZeros k
  let sums ← if (grayCode k).testBit j then tick (s.sums + 2 • (fun i => natIdx (A i) j)) (2 * (n + 1))
    else tick (s.sums - 2 • (fun i => natIdx (A i) j)) (2 * (n + 1))
  let sign ← tick (-s.sign) 1
  let p ← bigProd univ fun i => pure (sums i)
  let t ← tick (sign * p) 1
  tick ⟨sums, sign, s.acc + t⟩ 1

/-- Instrumented `ryserShiftGray`. -/
def ryserShiftGrayC {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) : Costed R := do
  let init ← vec fun i => do
    let rs ← bigSum univ fun j => pure (A i j)
    tick (2 * A i (Fin.last n) - rs) 2
  let sign0 ← tick ((-1 : R) ^ n) n
  let p0 ← bigProd univ fun i => pure (init i)
  let a0 ← tick (sign0 * p0) 1
  let st ← loop (ryserShiftGrayStepC A) ⟨init, sign0, a0⟩ (2 ^ n - 1)
  pure st.acc

section Field

variable {K : Type*} [Field K]

/-- Instrumented Nijenhuis–Wilf formula (every subset's sums from scratch). -/
def ryserNWC {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K) : Costed K := do
  let s ← bigSum univ fun S : Finset (Fin n) => do
    let p ← bigProd univ fun i => do
      let rs ← bigSum univ fun j => pure (A i j)
      let c ← bigSum S fun j => pure (A i j.castSucc)
      tick (A i (Fin.last n) - rs / 2 + c) 3
    tick ((-1) ^ #S * p) 1
  tick ((-1) ^ n * 2 * s) 2

/-- Instrumented normalized BB/FG: `(n+1)²` divisions, `n+1` multiplications for
`∏ r i`, the `glynnWiki` loop, and one final multiplication. -/
def glynnNormC {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K) (r : Fin (n + 1) → K) :
    Costed K := do
  let B ← vec fun i => vec fun j => tick (A i j / r i) 1
  let pr ← bigProd univ fun i => pure (r i)
  let g ← glynnWikiC (Matrix.of B)
  tick (pr * g) 1

end Field

end MaxelerPermanent

-- END DEFINITIONS

namespace MaxelerPermanent

variable {R : Type*} [CommRing R]

/-- The textbook sum over permutations is the permanent. -/
theorem permNaive_correct {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    permNaive A = A.permanent := by
  sorry

/-- The Leibniz sum is the determinant. -/
theorem detNaive_correct {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    detNaive A = A.det := by
  sorry

/-- Ryser's formula. -/
theorem ryser_correct {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    ryser A = A.permanent := by
  sorry

/-- The BB/FG (Glynn) formula with `δ₀ = +1`. -/
theorem glynn_correct {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynn A = 2 ^ n * A.permanent := by
  sorry

/-- Ryser's formula in binary Gray-code order (Nijenhuis–Wilf). -/
theorem ryserGray_correct {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    ryserGray A = A.permanent := by
  sorry

/-- BB/FG in binary Gray-code order. -/
theorem glynnGray_correct {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynnGray A = 2 ^ n * A.permanent := by
  sorry

/-- Ryser's formula with row and column multiplicities. -/
theorem ryserRep_correct {n p q : ℕ} (A : Matrix (Fin p) (Fin q) R)
    (r : Fin n → Fin p) (c : Fin n → Fin q) :
    (A.submatrix r c).permanent = ryserRep A (mult r) (mult c) := by
  sorry

/-- BB/FG with row and column multiplicities; the repeated row `0` is the copy with
fixed sign `+1`. -/
theorem glynnRep_correct {n p q : ℕ} (A : Matrix (Fin p) (Fin q) R)
    (r : Fin (n + 1) → Fin p) (c : Fin (n + 1) → Fin q) :
    2 ^ n * (A.submatrix r c).permanent =
      glynnRep A (r 0) (mult (r ∘ Fin.succ)) (mult c) := by
  sorry

/-- Ryser with multiplicities in n-ary Gray-code order. -/
theorem ryserRepGray_correct {n p q : ℕ} (A : Matrix (Fin p) (Fin q) R)
    (r : Fin n → Fin p) (c : Fin n → Fin q) :
    (A.submatrix r c).permanent = ryserRepGray A (mult r) (mult c) := by
  sorry

/-- BB/FG with multiplicities in n-ary Gray-code order, as in the generalised
Clifford & Clifford sampler. -/
theorem glynnRepGray_correct {n p q : ℕ} (A : Matrix (Fin p) (Fin q) R)
    (r : Fin (n + 1) → Fin p) (c : Fin (n + 1) → Fin q) :
    2 ^ n * (A.submatrix r c).permanent =
      glynnRepGray A (r 0) (mult (r ∘ Fin.succ)) (mult c) := by
  sorry

/-- BB/FG in the Gray-code form of `permanent_benchmark.py` / Wikipedia (last row fixed,
signs alternating with the step counter). -/
theorem glynnWiki_correct {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynnWiki A = 2 ^ n * A.permanent := by
  sorry

/-- Naive permanent: `n! (n + 1)` ring operations. -/
theorem permNaive_complexity {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    permNaiveC A = ⟨permNaive A, n.factorial * (n + 1)⟩ := by
  sorry

/-- Leibniz determinant: `n! (n + 2)` ring operations. -/
theorem detNaive_complexity {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    detNaiveC A = ⟨detNaive A, n.factorial * (n + 2)⟩ := by
  sorry

/-- Direct Ryser: `Θ(n² 2ⁿ)` ring operations. -/
theorem ryser_complexity {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    ryserC A = ⟨ryser A, n ^ 2 * 2 ^ (n - 1) + (n + 2) * 2 ^ n⟩ := by
  sorry

/-- Direct BB/FG on an `(n+1) × (n+1)` matrix: `Θ(n² 2ⁿ)` ring operations. -/
theorem glynn_complexity {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynnC A = ⟨glynn A, 2 ^ n * ((n + 1) * (2 * n + 2) + 2)⟩ := by
  sorry

/-- Ryser in Gray-code order: `Θ(n 2ⁿ)` ring operations. -/
theorem ryserGray_complexity {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    ryserGrayC A = ⟨ryserGray A, (2 ^ n - 1) * (2 * n + 3) + n + 1⟩ := by
  sorry

/-- BB/FG in Gray-code order: `Θ(n 2ⁿ)` ring operations for size `n + 1`. -/
theorem glynnGray_complexity {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynnGrayC A = ⟨glynnGray A, (2 ^ n - 1) * (3 * n + 6) + (n + 1) ^ 2 + (n + 1)⟩ := by
  sorry

/-- `permanent_benchmark.py` BB/FG: `Θ(n 2ⁿ)` ring operations for size `n + 1`. -/
theorem glynnWiki_complexity {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynnWikiC A = ⟨glynnWiki A, (2 ^ n - 1) * (3 * n + 4) + (n + 1) ^ 2 + (n + 1)⟩ := by
  sorry

/-- Ryser with multiplicities, direct: `Θ((pq + |M|) ∏ (N_l + 1))`. -/
theorem ryserRep_complexity {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (M : Fin p → ℕ)
    (N : Fin q → ℕ) :
    ryserRepC A M N = ⟨ryserRep A M N,
      (∏ l, (N l + 1)) * (2 * p * q + ∑ k, M k + p + 3)⟩ := by
  sorry

/-- BB/FG with multiplicities, direct: `Θ((pq + |N|) ∏ (M_k + 1))`. -/
theorem glynnRep_complexity {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (k0 : Fin p)
    (M : Fin p → ℕ) (N : Fin q → ℕ) :
    glynnRepC A k0 M N = ⟨glynnRep A k0 M N,
      (∏ k, (M k + 1)) * (2 * p * q + 2 * q + ∑ l, N l + 3)⟩ := by
  sorry

/-- Ryser with multiplicities, n-ary Gray code: `Θ((p + |M|) ∏ (N_l + 1))`. -/
theorem ryserRepGray_complexity {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (M : Fin p → ℕ)
    (N : Fin q → ℕ) :
    ryserRepGrayC A M N = ⟨ryserRepGray A M N,
      (∏ l, (N l + 1) - 1) * (2 * p + ∑ k, M k + 4) + ∑ k, M k + p + 1⟩ := by
  sorry

/-- BB/FG with multiplicities, n-ary Gray code: `Θ((q + |N|) ∏ (M_k + 1))`. -/
theorem glynnRepGray_complexity {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (k0 : Fin p)
    (M : Fin p → ℕ) (N : Fin q → ℕ) :
    glynnRepGrayC A k0 M N = ⟨glynnRepGray A k0 M N,
      (∏ k, (M k + 1) - 1) * (3 * q + ∑ l, N l + 4) + q * (2 * p + 1) + ∑ l, N l + q⟩ := by
  sorry

/-- Shifted Ryser (Nijenhuis–Wilf row-sum shift, scaled by `2`) on an `(n+1) × (n+1)` matrix. -/
theorem ryserShift_correct {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    ryserShift A = 2 ^ n * A.permanent := by
  sorry

/-- Shifted Ryser in binary Gray-code order (the traditional Nijenhuis–Wilf loop). -/
theorem ryserShiftGray_correct {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    ryserShiftGray A = 2 ^ n * A.permanent := by
  sorry

/-- Shifted Ryser, direct: `Θ(n² 2ⁿ)` ring operations. -/
theorem ryserShift_complexity {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    ryserShiftC A = ⟨ryserShift A, 2 ^ n * ((n + 1) * (n + 6) + 2) + (n + 1) * (n * 2 ^ (n - 1))⟩ := by
  sorry

/-- Shifted Ryser in Gray-code order: `Θ(n 2ⁿ)` ring operations for size `n + 1`. -/
theorem ryserShiftGray_complexity {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    ryserShiftGrayC A = ⟨ryserShiftGray A, (2 ^ n - 1) * (3 * n + 6) + (n + 1) * (n + 5)⟩ := by
  sorry

section Field

variable {K : Type*} [Field K]

/-- The Nijenhuis–Wilf formula as published (with halves), over a field with `2 ≠ 0`. -/
theorem ryserNW_correct {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K) (h2 : (2 : K) ≠ 0) :
    ryserNW A = A.permanent := by
  sorry

/-- Normalized BB/FG (`permanent_glynn_gray_fixpt`): scale rows by `r i ≠ 0`, run the Gray-code loop, multiply back. -/
theorem glynnNorm_correct {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K) (r : Fin (n + 1) → K)
    (hr : ∀ i, r i ≠ 0) :
    glynnNorm A r = 2 ^ n * A.permanent := by
  sorry

/-- Nijenhuis–Wilf formula, direct: `Θ(n² 2ⁿ)` field operations. -/
theorem ryserNW_complexity {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K) :
    ryserNWC A =
      ⟨ryserNW A, 2 ^ n * ((n + 1) * (n + 5) + 2) + (n + 1) * (n * 2 ^ (n - 1)) + 2⟩ := by
  sorry

/-- Normalized BB/FG: `(n+1)²` divisions plus the `Θ(n 2ⁿ)` loop. -/
theorem glynnNorm_complexity {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K) (r : Fin (n + 1) → K) :
    glynnNormC A r = ⟨glynnNorm A r,
      (2 ^ n - 1) * (3 * n + 4) + 2 * (n + 1) ^ 2 + 2 * (n + 1) + 1⟩ := by
  sorry

end Field

end MaxelerPermanent
