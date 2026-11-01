module

public import MaxelerPermanent.Basic

/-!
# Repeated rows and columns

For selections `r c : Fin n → Fin p` the matrix `A.submatrix r c` repeats row `k`
`mult r k` times and column `l` `mult c l` times. The multiplicity formulas group
the `2ⁿ` subsets of Ryser / sign vectors of BB/FG by how many elements they pick
from each block of repeated rows/columns, and weight each group by its size
`∏ C(M_k, s_k)` (`card_filter_counts`).

* `ryserRep_eq_permanent`: `perm (A.submatrix r c) = ryserRep A (mult r) (mult c)`.
* `glynnRep_eq`: `2ⁿ · perm (A.submatrix r c) = glynnRep A (r 0) (mult (r ∘ succ)) (mult c)`.
-/

public section

open Finset

namespace MaxelerPermanent

variable {R : Type*} [CommRing R]

/-- How many elements of `S` lie over `l`. -/
def counts {n q : ℕ} (c : Fin n → Fin q) (S : Finset (Fin n)) (l : Fin q) : ℕ :=
  #{j ∈ S | c j = l}

theorem sum_mult {n q : ℕ} (c : Fin n → Fin q) : ∑ l, mult c l = n := by
  have := (card_eq_sum_card_fiberwise (s := univ) (t := univ) (f := c)
    (fun _ _ => mem_univ _)).symm
  simpa [mult] using this

theorem sum_counts {n q : ℕ} (c : Fin n → Fin q) (S : Finset (Fin n)) :
    ∑ l, counts c S l = #S := by
  have := (card_eq_sum_card_fiberwise (s := S) (t := univ) (f := c)
    (fun _ _ => mem_univ _)).symm
  simpa [counts] using this

theorem counts_le_mult {n q : ℕ} (c : Fin n → Fin q) (S : Finset (Fin n)) (l : Fin q) :
    counts c S l ≤ mult c l := by
  apply card_le_card
  intro j
  simp only [mem_filter, mem_univ, true_and]
  exact fun h => h.2

/-- The number of subsets with prescribed counts in each block is `∏ C(M_l, s_l)`. -/
theorem card_filter_counts {n q : ℕ} (c : Fin n → Fin q) (s : Fin q → ℕ) :
    #{S : Finset (Fin n) | counts c S = s} = ∏ l, (mult c l).choose (s l) := by
  let blk : Fin q → Finset (Fin n) := fun l => {j | c j = l}
  have hcard : ∏ l, (mult c l).choose (s l) =
      #(Fintype.piFinset fun l => powersetCard (s l) (blk l)) := by
    rw [Fintype.card_piFinset]
    refine prod_congr rfl fun l _ => ?_
    rw [card_powersetCard]
    rfl
  rw [hcard]
  have key : ∀ T : Fin q → Finset (Fin n), (∀ l, T l ⊆ blk l) → ∀ l,
      (univ.filter fun j => j ∈ T (c j)).filter (c · = l) = T l := by
    intro T hT l
    ext j
    simp only [mem_filter, mem_univ, true_and]
    constructor
    · rintro ⟨h, rfl⟩; exact h
    · intro h
      have := hT l h
      simp only [blk, mem_filter, mem_univ, true_and] at this
      exact ⟨this ▸ h, this⟩
  refine card_nbij' (fun S l => S.filter (c · = l))
    (fun T => univ.filter fun j => j ∈ T (c j)) ?_ ?_ ?_ ?_
  · intro S hS
    simp only [mem_coe, mem_filter, mem_univ, true_and] at hS
    simp only [mem_coe, Fintype.mem_piFinset, mem_powersetCard]
    intro l
    refine ⟨?_, ?_⟩
    · intro j; simp [blk]
    · rw [← hS]; rfl
  · intro T hT
    simp only [mem_coe, Fintype.mem_piFinset, mem_powersetCard] at hT
    simp only [mem_coe, mem_filter, mem_univ, true_and]
    funext l
    rw [← (hT l).2, counts]
    exact congrArg card (key T (fun l => (hT l).1) l)
  · intro S _
    ext j; simp
  · intro T hT
    simp only [mem_coe, Fintype.mem_piFinset, mem_powersetCard] at hT
    funext l
    exact key T (fun l => (hT l).1) l

/-- Grouping a sum over subsets by block counts. -/
theorem sum_finset_eq_sum_counts {n q : ℕ} {M : Type*} [AddCommMonoid M]
    (c : Fin n → Fin q) (G : (Fin q → ℕ) → M) :
    ∑ S : Finset (Fin n), G (counts c S) =
      ∑ s ∈ Fintype.piFinset (fun l => range (mult c l + 1)),
        (∏ l, (mult c l).choose (s l)) • G s := by
  rw [← sum_fiberwise_of_maps_to (g := counts c)
    (t := Fintype.piFinset (fun l => range (mult c l + 1)))]
  · refine sum_congr rfl fun s _ => ?_
    rw [← card_filter_counts, ← sum_const]
    refine sum_congr rfl fun S hS => ?_
    rw [(mem_filter.mp hS).2]
  · intro S _
    simp only [Fintype.mem_piFinset, mem_range]
    exact fun l => Nat.lt_succ_of_le (counts_le_mult c S l)

/-- `∑_{j ∈ S} F (c j) = ∑_l counts c S l • F l`. -/
theorem sum_comp_eq_sum_counts {n q : ℕ} {M : Type*} [AddCommMonoid M]
    (c : Fin n → Fin q) (S : Finset (Fin n)) (F : Fin q → M) :
    ∑ j ∈ S, F (c j) = ∑ l, counts c S l • F l := by
  rw [← sum_fiberwise' S c F]
  refine sum_congr rfl fun l _ => ?_
  rw [sum_const]
  rfl

/-- `∏_i F (r i) = ∏_k F k ^ mult r k`. -/
theorem prod_comp_eq_prod_mult {n p : ℕ} {M : Type*} [CommMonoid M]
    (r : Fin n → Fin p) (F : Fin p → M) :
    ∏ i, F (r i) = ∏ k, F k ^ mult r k := by
  rw [← prod_fiberwise' univ r F]
  refine prod_congr rfl fun k _ => ?_
  rw [prod_const]
  rfl

theorem neg_one_pow_sub {a b : ℕ} (h : b ≤ a) :
    (-1 : R) ^ (a - b) = (-1) ^ a * (-1) ^ b := by
  have e : a = (a - b) + b := by omega
  conv_rhs => rw [e, pow_add, mul_assoc, ← pow_add, ← two_mul, pow_mul]
  simp

/-- **Ryser with multiplicities.** -/
theorem ryserRep_eq_permanent {n p q : ℕ} (A : Matrix (Fin p) (Fin q) R)
    (r : Fin n → Fin p) (c : Fin n → Fin q) :
    (A.submatrix r c).permanent = ryserRep A (mult r) (mult c) := by
  rw [← ryser_eq_permanent, ryser]
  have hterm : ∀ S : Finset (Fin n),
      (-1 : R) ^ (n - #S) * ∏ i, ∑ j ∈ S, A.submatrix r c i j =
        (-1) ^ (∑ l, mult c l - ∑ l, counts c S l) *
          ∏ k, (∑ l, (counts c S l : R) * A k l) ^ mult r k := by
    intro S
    rw [sum_mult, sum_counts]
    congr 1
    simp only [Matrix.submatrix_apply]
    rw [prod_comp_eq_prod_mult r (fun k => ∑ j ∈ S, A k (c j))]
    refine prod_congr rfl fun k _ => ?_
    rw [sum_comp_eq_sum_counts c S (fun l => A k l)]
    simp [nsmul_eq_mul]
  simp_rw [hterm]
  rw [sum_finset_eq_sum_counts c (fun s => (-1 : R) ^ (∑ l, mult c l - ∑ l, s l) *
      ∏ k, (∑ l, (s l : R) * A k l) ^ mult r k), ryserRep]
  refine sum_congr rfl fun s _ => ?_
  rw [nsmul_eq_mul]
  ring

/-- `∑_{i ∈ block k} δ_i = M_k - 2 d_k` for the block of `k` under `r`. -/
theorem sum_flipSign_block {n p : ℕ} (r : Fin n → Fin p) (S : Finset (Fin n)) (k : Fin p) :
    ∑ i ∈ univ.filter (r · = k), (flipSign S i : R) =
      (mult r k : R) - 2 * (counts r S k : R) := by
  simp only [flipSign]
  rw [sum_ite, sum_const, sum_const, nsmul_eq_mul, nsmul_eq_mul, mul_neg, mul_one, mul_one]
  have hsplit := card_filter_add_card_filter_not (s := univ.filter (r · = k)) (· ∈ S)
  have hc : #((univ.filter (r · = k)).filter (· ∈ S)) = counts r S k := by
    rw [counts]; congr 1; ext j; simp [and_comm]
  rw [hc] at hsplit
  have hm : (mult r k : R) = (counts r S k : R) +
      (#((univ.filter (r · = k)).filter (fun j => j ∉ S)) : R) := by
    rw [mult]; exact_mod_cast congrArg (Nat.cast : ℕ → R) hsplit.symm
  rw [hc, hm]
  ring

theorem sum_flipSign_mul_comp {n p : ℕ} (r : Fin n → Fin p) (S : Finset (Fin n))
    (F : Fin p → R) :
    ∑ i, flipSign S i * F (r i) = ∑ k, ((mult r k : R) - 2 * (counts r S k : R)) * F k := by
  rw [← sum_fiberwise univ r]
  refine sum_congr rfl fun k _ => ?_
  rw [← sum_flipSign_block, sum_mul]
  refine sum_congr rfl fun i hi => ?_
  rw [(mem_filter.mp hi).2]

/-- **BB/FG with multiplicities.** Row `0` of the repeated matrix is the copy with
fixed sign `+1`; the remaining rows `r ∘ Fin.succ` carry the multiplicities. -/
theorem glynnRep_eq {n p q : ℕ} (A : Matrix (Fin p) (Fin q) R)
    (r : Fin (n + 1) → Fin p) (c : Fin (n + 1) → Fin q) :
    2 ^ n * (A.submatrix r c).permanent =
      glynnRep A (r 0) (mult (r ∘ Fin.succ)) (mult c) := by
  rw [← glynn_eq, glynn]
  set r' := r ∘ Fin.succ
  have hterm : ∀ S : Finset (Fin n),
      (-1 : R) ^ #S * ∏ j, (A.submatrix r c 0 j +
          ∑ i : Fin n, flipSign S i * A.submatrix r c i.succ j) =
        (-1) ^ (∑ k, counts r' S k) *
          ∏ l, (A (r 0) l + ∑ k, ((mult r' k : R) - 2 * (counts r' S k : R)) * A k l) ^
            mult c l := by
    intro S
    rw [sum_counts]
    congr 1
    simp only [Matrix.submatrix_apply]
    rw [prod_comp_eq_prod_mult c (fun l => A (r 0) l +
      ∑ i : Fin n, flipSign S i * A (r i.succ) l)]
    refine prod_congr rfl fun l _ => ?_
    congr 2
    exact sum_flipSign_mul_comp r' S (fun k => A k l)
  simp_rw [hterm]
  rw [sum_finset_eq_sum_counts r' (fun d => (-1 : R) ^ (∑ k, d k) *
      ∏ l, (A (r 0) l + ∑ k, ((mult r' k : R) - 2 * (d k : R)) * A k l) ^ mult c l),
    glynnRep]
  refine sum_congr rfl fun d _ => ?_
  rw [nsmul_eq_mul]
  ring

/-! ## The canonical repeated matrix -/

/-- The canonical expansion map `Fin (∑ M) → Fin p` sending the `M k` copies of `k`
to `k` (copies of row `0` first, then of row `1`, …). -/
def expand {p : ℕ} (M : Fin p → ℕ) (i : Fin (∑ k, M k)) : Fin p :=
  (finSigmaFinEquiv.symm i).1

theorem mult_expand {p : ℕ} (M : Fin p → ℕ) (k : Fin p) : mult (expand M) k = M k := by
  rw [mult]
  have : (univ.filter fun i : Fin (∑ k, M k) => expand M i = k) =
      (univ.filter fun x : (j : Fin p) × Fin (M j) => x.1 = k).map
        finSigmaFinEquiv.toEmbedding := by
    ext i
    simp only [mem_filter, mem_univ, true_and, mem_map_equiv]
    rfl
  rw [this, card_map]
  rw [show (univ.filter fun x : (j : Fin p) × Fin (M j) => x.1 = k) =
    (univ : Finset (Fin (M k))).map (Function.Embedding.sigmaMk k) by
      ext ⟨j, x⟩
      simp only [mem_filter, mem_univ, true_and, mem_map, Function.Embedding.sigmaMk_apply]
      constructor
      · rintro rfl; exact ⟨x, rfl⟩
      · rintro ⟨y, h⟩; exact (congrArg Sigma.fst h).symm]
  simp

/-- The matrix with row `k` repeated `M k` times and column `l` repeated `N l` times. -/
def repeatMatrix {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (M : Fin p → ℕ) (N : Fin q → ℕ)
    (h : ∑ l, N l = ∑ k, M k) : Matrix (Fin (∑ k, M k)) (Fin (∑ k, M k)) R :=
  A.submatrix (expand M) (fun j => expand N (Fin.cast h.symm j))

theorem ryserRep_eq_permanent_repeatMatrix {p q : ℕ} (A : Matrix (Fin p) (Fin q) R)
    (M : Fin p → ℕ) (N : Fin q → ℕ) (h : ∑ l, N l = ∑ k, M k) :
    (repeatMatrix A M N h).permanent = ryserRep A M N := by
  rw [repeatMatrix, ryserRep_eq_permanent]
  congr 1
  · funext k; exact mult_expand M k
  · funext l
    rw [← mult_expand N l, mult, mult]
    refine card_nbij' (Fin.cast h.symm) (Fin.cast h) ?_ ?_ ?_ ?_ <;> intro x hx <;>
      simp_all [Fin.cast]

end MaxelerPermanent
