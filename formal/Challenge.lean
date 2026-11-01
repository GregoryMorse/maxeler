module

public import Mathlib.LinearAlgebra.Matrix.Permanent
public import Mathlib.LinearAlgebra.Matrix.Determinant.Basic

/-!
# Advertised statements

Correctness of exact permanent algorithms used for boson sampling simulation in
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

end MaxelerPermanent
