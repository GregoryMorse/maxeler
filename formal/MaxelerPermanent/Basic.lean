module

public import MaxelerPermanent.Defs
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Algebra.BigOperators.GroupWithZero.Finset
public import Mathlib.Data.Nat.Choose.Sum
public import Mathlib.Data.Fintype.BigOperators
public import Mathlib.Algebra.Ring.Parity

/-!
# Naive formulas, Ryser and BB/FG (Glynn)

* `permNaive_eq_permanent`, `detNaive_eq_det`: the textbook sums agree with
  Mathlib's `Matrix.permanent` and `Matrix.det`.
* `ryser_eq_permanent`: Ryser's formula.
* `glynn_eq`: the BB/FG formula, `glynn A = 2ⁿ · perm A`.
-/

public section

open Finset

namespace MaxelerPermanent

variable {R : Type*} [CommRing R]

/-! ## Naive formulas -/

theorem permNaive_eq_permanent {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    permNaive A = A.permanent := by
  rw [← Matrix.permanent_transpose]
  rfl

theorem detNaive_eq_det {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    detNaive A = A.det := by
  rw [← Matrix.det_transpose, Matrix.det_apply]
  refine sum_congr rfl fun σ _ => ?_
  rw [Units.smul_def, zsmul_eq_mul]
  rfl

/-! ## Sums over surjective maps -/

/-- Summing over the surjective self-maps of `Fin n` is summing over permutations. -/
theorem sum_surjective_eq_sum_perm {n : ℕ} {M : Type*} [AddCommMonoid M]
    (g : (Fin n → Fin n) → M) :
    ∑ f ∈ univ.filter (fun f : Fin n → Fin n => Function.Surjective f), g f =
      ∑ σ : Equiv.Perm (Fin n), g σ := by
  symm
  refine sum_bij (fun σ _ => (σ : Fin n → Fin n)) ?_ ?_ ?_ ?_
  · intro σ _
    simpa using σ.surjective
  · intro a _ b _ h
    exact Equiv.coe_fn_injective h
  · intro f hf
    simp only [mem_filter, mem_univ, true_and] at hf
    have hb : Function.Bijective f := ⟨Finite.injective_iff_surjective.mpr hf, hf⟩
    exact ⟨Equiv.ofBijective f hb, mem_univ _, rfl⟩
  · intro _ _
    rfl

/-! ## Ryser -/

/-- The alternating sum over the supersets `S ⊇ T` of `(-1)^(n - |S|)`. -/
theorem sum_superset_neg_one_pow {n : ℕ} (T : Finset (Fin n)) :
    ∑ S : Finset (Fin n), (if T ⊆ S then (-1 : R) ^ (n - #S) else 0) =
      if T = univ then 1 else 0 := by
  have key : ∑ S : Finset (Fin n), (if T ⊆ S then (-1 : R) ^ (n - #S) else 0) =
      ∑ C ∈ (Tᶜ).powerset, (-1 : R) ^ #C := by
    rw [← sum_filter]
    refine sum_nbij' (fun S => Sᶜ) (fun C => Cᶜ) ?_ ?_ ?_ ?_ ?_
    · intro S hS
      simp only [mem_filter, mem_univ, true_and, mem_powerset] at hS ⊢
      exact compl_subset_compl.mpr hS
    · intro C hC
      simp only [mem_filter, mem_univ, true_and, mem_powerset] at hC ⊢
      exact subset_compl_comm.mp hC
    · intro S _; simp
    · intro C _; simp
    · intro S _
      have hS := card_le_univ S
      rw [Fintype.card_fin] at hS
      rw [card_compl, Fintype.card_fin]
  rw [key]
  have h := sum_powerset_neg_one_pow_card (x := Tᶜ)
  have h' := congrArg (Int.cast : ℤ → R) h
  push_cast at h'
  rw [h']
  simp [compl_eq_empty_iff]

theorem ryser_eq_permanent {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    ryser A = A.permanent := by
  rw [← permNaive_eq_permanent, permNaive, ryser]
  -- expand the product of row sums into a sum over maps `f : Fin n → Fin n`
  have expand : ∀ S : Finset (Fin n), ∏ i, ∑ j ∈ S, A i j =
      ∑ f : Fin n → Fin n, if univ.image f ⊆ S then ∏ i, A i (f i) else 0 := by
    intro S
    rw [prod_univ_sum (fun _ => S) (fun i j => A i j), ← sum_filter]
    refine sum_congr ?_ fun _ _ => rfl
    ext f
    simp [Fintype.mem_piFinset, image_subset_iff]
  simp_rw [expand, mul_sum]
  rw [sum_comm]
  have inner : ∀ f : Fin n → Fin n,
      ∑ S : Finset (Fin n), (-1 : R) ^ (n - #S) *
          (if univ.image f ⊆ S then ∏ i, A i (f i) else 0) =
        if Function.Surjective f then ∏ i, A i (f i) else 0 := by
    intro f
    have := sum_superset_neg_one_pow (R := R) (univ.image f)
    simp_rw [mul_ite, mul_zero]
    calc ∑ S : Finset (Fin n), (if univ.image f ⊆ S then
            (-1 : R) ^ (n - #S) * ∏ i, A i (f i) else 0)
        = (∑ S : Finset (Fin n), if univ.image f ⊆ S then (-1 : R) ^ (n - #S) else 0) *
            ∏ i, A i (f i) := by
          rw [sum_mul]
          refine sum_congr rfl fun S _ => ?_
          split_ifs <;> simp
      _ = _ := by
          rw [this]
          have : univ.image f = univ ↔ Function.Surjective f := by
            rw [← coe_inj, coe_image, coe_univ, Set.image_univ, Set.range_eq_univ]
          by_cases hf : Function.Surjective f
          · simp [this.mpr hf, hf]
          · simp [mt this.mp hf, hf]
  simp_rw [inner]
  rw [← sum_filter]
  exact sum_surjective_eq_sum_perm (fun f => ∏ i, A i (f i))

/-! ## BB/FG (Glynn) -/

/-- Size of the fibre of `f` over `k`. -/
def fiber {n : ℕ} (f : Fin n → Fin n) (k : Fin n) : ℕ := #{j | f j = k}

theorem prod_comp_eq_prod_pow {n : ℕ} {M : Type*} [CommMonoid M] (f : Fin n → Fin n)
    (e : Fin n → M) : ∏ j, e (f j) = ∏ k, e k ^ fiber f k := by
  rw [← prod_fiberwise' univ f e]
  refine prod_congr rfl fun k _ => ?_
  rw [prod_const]
  rfl

theorem sum_fiber {n : ℕ} (f : Fin n → Fin n) : ∑ k, fiber f k = n := by
  have := (card_eq_sum_card_fiberwise (s := univ) (t := univ) (f := f)
    (fun _ _ => mem_univ _)).symm
  simpa [fiber] using this

theorem fiber_eq_one_of_surjective {n : ℕ} {f : Fin n → Fin n} (hf : Function.Surjective f)
    (k : Fin n) : fiber f k = 1 := by
  have hb : Function.Bijective f := ⟨Finite.injective_iff_surjective.mpr hf, hf⟩
  obtain ⟨j, hj⟩ := hf k
  rw [fiber, card_eq_one]
  refine ⟨j, ?_⟩
  ext i
  simp only [mem_filter, mem_univ, true_and, mem_singleton]
  constructor
  · intro hi; exact hb.1 (hi.trans hj.symm)
  · rintro rfl; exact hj

theorem surjective_of_fiber_pos {n : ℕ} {f : Fin n → Fin n} (h : ∀ k, 0 < fiber f k) :
    Function.Surjective f := by
  intro k
  obtain ⟨j, hj⟩ := card_pos.mp (h k)
  exact ⟨j, (mem_filter.mp hj).2⟩

/-- With the fibres over all `k ≠ 0` odd, the fibre over `0` is odd as well. -/
theorem odd_fiber_zero {n : ℕ} (f : Fin (n + 1) → Fin (n + 1))
    (h : ∀ i : Fin n, Odd (fiber f i.succ)) : Odd (fiber f 0) := by
  have hsum := sum_fiber f
  rw [Fin.sum_univ_succ] at hsum
  have hprod : ∏ i : Fin n, ((-1 : ℤ) ^ fiber f i.succ) = (-1) ^ n := by
    rw [prod_congr rfl fun i _ => (h i).neg_one_pow]
    simp
  have key : (-1 : ℤ) ^ (n + 1) = (-1) ^ fiber f 0 * (-1) ^ n := by
    conv_lhs => rw [← hsum]
    rw [pow_add, ← prod_pow_eq_pow_sum, hprod]
  have : (-1 : ℤ) ^ fiber f 0 = -1 := by
    have h1 : ((-1 : ℤ) ^ n) ^ 2 = 1 := by rw [← pow_mul, mul_comm, pow_mul]; simp
    calc (-1 : ℤ) ^ fiber f 0 = (-1) ^ fiber f 0 * ((-1) ^ n) ^ 2 := by rw [h1, mul_one]
      _ = (-1) ^ (n + 1) * (-1) ^ n := by rw [key]; ring
      _ = -1 := by rw [pow_succ, mul_assoc, mul_comm (-1 : ℤ), ← mul_assoc, ← sq, h1]; ring
  exact (neg_one_pow_eq_neg_one_iff_odd (by decide)).mp this

/-- `∑_{S ⊆ [n]} ∏_i (δ_i)^(c_i + 1) = ∏_i (1 + (-1)^(c_i + 1))`. -/
theorem sum_prod_flipSign {n : ℕ} (c : Fin n → ℕ) :
    ∑ S : Finset (Fin n), ∏ i, (flipSign S i : R) ^ (c i + 1) =
      if ∀ i, Odd (c i) then 2 ^ n else 0 := by
  have hS : ∀ S : Finset (Fin n), ∏ i, (flipSign S i : R) ^ (c i + 1) =
      (∏ i ∈ S, (-1 : R) ^ (c i + 1)) * ∏ i ∈ univ \ S, (1 : R) := by
    intro S
    simp only [prod_const_one, mul_one, flipSign]
    rw [← Fintype.prod_ite_mem S]
    refine prod_congr rfl fun i _ => ?_
    by_cases hi : i ∈ S <;> simp [hi]
  simp_rw [hS]
  rw [← powerset_univ, ← prod_add]
  have : ∀ i, ((-1 : R) ^ (c i + 1) + 1) = if Odd (c i) then 2 else 0 := by
    intro i
    rcases Nat.even_or_odd (c i) with h | h
    · rw [if_neg (Nat.not_odd_iff_even.mpr h), (h.add_one).neg_one_pow]; ring
    · rw [if_pos h, (h.add_one).neg_one_pow]; ring
  simp_rw [this]
  rw [prod_ite_zero, prod_const, card_univ, Fintype.card_fin]
  simp

theorem neg_one_pow_card_eq_prod_flipSign {n : ℕ} (S : Finset (Fin n)) :
    (-1 : R) ^ #S = ∏ i, flipSign S i := by
  simp only [flipSign]
  rw [show (∏ i, if i ∈ S then (-1 : R) else 1) = ∏ i ∈ S, (-1 : R) from
    Fintype.prod_ite_mem S _, prod_const]

/-- **BB/FG formula.** For an `(n+1) × (n+1)` matrix, the Glynn sum with `δ₀ = +1`
equals `2ⁿ` times the permanent. -/
theorem glynn_eq {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynn A = 2 ^ n * A.permanent := by
  -- `e S` is the full sign vector, with `e S 0 = 1`
  let e : Finset (Fin n) → Fin (n + 1) → R := fun S => Fin.cons 1 (flipSign S)
  have hcol : ∀ S : Finset (Fin n), ∀ j,
      A 0 j + ∑ i : Fin n, flipSign S i * A i.succ j = ∑ i, e S i * A i j := by
    intro S j
    rw [Fin.sum_univ_succ]
    simp [e]
  have expand : ∀ S : Finset (Fin n), ∏ j, ∑ i, e S i * A i j =
      ∑ f : Fin (n + 1) → Fin (n + 1), (∏ j, e S (f j)) * ∏ j, A (f j) j := by
    intro S
    rw [prod_univ_sum]
    simp [prod_mul_distrib]
  simp_rw [glynn, hcol, expand, mul_sum]
  rw [sum_comm]
  have weight : ∀ f : Fin (n + 1) → Fin (n + 1),
      ∑ S : Finset (Fin n), (-1 : R) ^ #S * ((∏ j, e S (f j)) * ∏ j, A (f j) j) =
        (if Function.Surjective f then 2 ^ n else 0) * ∏ j, A (f j) j := by
    intro f
    simp_rw [← mul_assoc]
    rw [← sum_mul]
    congr 1
    have h1 : ∀ S : Finset (Fin n), (-1 : R) ^ #S * ∏ j, e S (f j) =
        ∏ i : Fin n, (flipSign S i : R) ^ (fiber f i.succ + 1) := by
      intro S
      rw [prod_comp_eq_prod_pow, Fin.prod_univ_succ, neg_one_pow_card_eq_prod_flipSign]
      simp only [e, Fin.cons_zero, one_pow, one_mul, Fin.cons_succ, pow_succ]
      rw [prod_mul_distrib, mul_comm]
    simp_rw [h1]
    rw [sum_prod_flipSign]
    congr 1
    apply propext
    constructor
    · intro h
      apply surjective_of_fiber_pos
      intro k
      refine Fin.cases ?_ (fun i => ?_) k
      · exact (odd_fiber_zero f h).pos
      · exact (h i).pos
    · intro hf i
      rw [fiber_eq_one_of_surjective hf]
      exact odd_one
  simp_rw [weight, ite_mul, zero_mul]
  rw [← sum_filter, ← mul_sum, sum_surjective_eq_sum_perm (fun f => ∏ j, A (f j) j)]
  rfl

end MaxelerPermanent
