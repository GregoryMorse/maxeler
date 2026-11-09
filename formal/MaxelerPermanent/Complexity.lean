module

public import MaxelerPermanent.Wiki

/-!
# Operation counts

For every algorithm, the instrumented program `…C` (see `Defs.lean` for the cost
model) computes exactly the algorithm's value, with an exactly determined number of
ring operations:

| algorithm | ring operations |
|---|---|
| `permNaive` | `n! (n + 1)` |
| `detNaive` | `n! (n + 2)` |
| `ryser` | `n² 2ⁿ⁻¹ + (n + 2) 2ⁿ` |
| `glynn` (size `n+1`) | `2ⁿ ((n+1)(2n+2) + 2)` |
| `ryserGray` | `(2ⁿ - 1)(2n + 3) + n + 1` |
| `glynnGray` (size `n+1`) | `(2ⁿ - 1)(3n + 6) + (n+1)² + (n+1)` |
| `glynnWiki` (size `n+1`) | `(2ⁿ - 1)(3n + 4) + (n+1)² + (n+1)` |
| `ryserShift` (size `n+1`) | `2ⁿ ((n+1)(n+6) + 2) + (n+1) n 2ⁿ⁻¹` |
| `ryserNW` (size `n+1`) | `2ⁿ ((n+1)(n+5) + 2) + (n+1) n 2ⁿ⁻¹ + 2` |
| `ryserShiftGray` (size `n+1`) | `(2ⁿ - 1)(3n + 6) + (n+1)(n+5)` |
| `glynnNorm` (size `n+1`) | `(2ⁿ - 1)(3n + 4) + 2(n+1)² + 2(n+1) + 1` |
| `ryserRep` | `K (2pq + |M| + p + 3)`, `K = ∏ (N_l + 1)` |
| `glynnRep` | `K (2pq + 2q + |N| + 3)`, `K = ∏ (M_k + 1)` |
| `ryserRepGray` | `(K - 1)(2p + |M| + 4) + |M| + p + 1` |
| `glynnRepGray` | `(K - 1)(3q + |N| + 4) + q(2p + 1) + |N| + q` |

So the Gray-code variants save a factor of `n` over the direct formulas:
`Θ(n 2ⁿ)` against `Θ(n² 2ⁿ)`, and `Θ((q + |N|) ∏(M_k + 1))` against
`Θ(pq ∏(M_k + 1))` with multiplicities (exact counts, so the `Θ` bounds are
immediate).
-/

universe u

public section

open Finset

namespace MaxelerPermanent

namespace Costed

variable {α β : Type u}

@[simp] theorem val_pure (a : α) : (pure a : Costed α).val = a := rfl
@[simp] theorem cost_pure (a : α) : (pure a : Costed α).cost = 0 := rfl
@[simp] theorem val_bind (x : Costed α) (f : α → Costed β) : (x >>= f).val = (f x.val).val := rfl
@[simp] theorem cost_bind (x : Costed α) (f : α → Costed β) :
    (x >>= f).cost = x.cost + (f x.val).cost := rfl
@[simp] theorem val_tick (a : α) (k : ℕ) : (tick a k).val = a := rfl
@[simp] theorem cost_tick (a : α) (k : ℕ) : (tick a k).cost = k := rfl
@[simp] theorem val_ite (c : Prop) [Decidable c] (x y : Costed α) :
    (if c then x else y).val = if c then x.val else y.val := by split <;> rfl
@[simp] theorem cost_ite (c : Prop) [Decidable c] (x y : Costed α) :
    (if c then x else y).cost = if c then x.cost else y.cost := by split <;> rfl
@[simp] theorem val_bigSum {ι : Type*} [AddCommMonoid α] (s : Finset ι) (f : ι → Costed α) :
    (bigSum s f).val = ∑ i ∈ s, (f i).val := rfl
@[simp] theorem cost_bigSum {ι : Type*} [AddCommMonoid α] (s : Finset ι) (f : ι → Costed α) :
    (bigSum s f).cost = (∑ i ∈ s, (f i).cost) + #s := rfl
@[simp] theorem val_bigProd {ι : Type*} [CommMonoid α] (s : Finset ι) (f : ι → Costed α) :
    (bigProd s f).val = ∏ i ∈ s, (f i).val := rfl
@[simp] theorem cost_bigProd {ι : Type*} [CommMonoid α] (s : Finset ι) (f : ι → Costed α) :
    (bigProd s f).cost = (∑ i ∈ s, (f i).cost) + #s := rfl
@[simp] theorem val_vec {ι : Type*} [Fintype ι] (f : ι → Costed α) :
    (vec f).val = fun i => (f i).val := rfl
@[simp] theorem cost_vec {ι : Type*} [Fintype ι] (f : ι → Costed α) :
    (vec f).cost = ∑ i, (f i).cost := rfl

theorem val_loop {σ : Type u} (step : ℕ → σ → Costed σ) (init : σ) (T : ℕ) :
    (loop step init T).val = runSteps (fun k s => (step k s).val) init T := by
  induction T with
  | zero => rfl
  | succ T ih => simp [loop, runSteps, ih]

theorem cost_loop {σ : Type u} (step : ℕ → σ → Costed σ) (init : σ) (T c : ℕ)
    (h : ∀ k s, (step k s).cost = c) : (loop step init T).cost = T * c := by
  induction T with
  | zero => simp [loop]
  | succ T ih => simp [loop, ih, h, add_mul]

end Costed

open Costed

variable {R : Type*} [CommRing R]

/-- `∑_{S ⊆ [n]} |S| = n 2ⁿ⁻¹`. -/
theorem sum_card_finset (n : ℕ) : ∑ S : Finset (Fin n), #S = n * 2 ^ (n - 1) := by
  have hcompl : ∑ S : Finset (Fin n), #S = ∑ S : Finset (Fin n), (n - #S) := by
    refine Fintype.sum_bijective compl compl_involutive.bijective _ _ fun S => ?_
    rw [card_compl, Fintype.card_fin]
    have := card_le_univ S; rw [Fintype.card_fin] at this; omega
  have h2 : 2 * ∑ S : Finset (Fin n), #S = n * 2 ^ n := by
    rw [two_mul]
    conv_lhs => arg 2; rw [hcompl]
    rw [← sum_add_distrib]
    have : ∀ S : Finset (Fin n), #S + (n - #S) = n := fun S => by
      have := card_le_univ S; rw [Fintype.card_fin] at this; omega
    simp only [this, sum_const, card_univ, Fintype.card_finset, Fintype.card_fin, smul_eq_mul]
    ring
  rcases n with _ | n
  · rw [zero_mul]
    exact sum_eq_zero fun S _ => card_eq_zero.mpr (Subsingleton.elim _ _)
  · rw [Nat.add_sub_cancel, pow_succ] at *
    have : 2 * ∑ S : Finset (Fin (n + 1)), #S = 2 * ((n + 1) * 2 ^ n) := by rw [h2]; ring
    omega

/-! ## Naive formulas -/

theorem permNaiveC_eq {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    permNaiveC A = ⟨permNaive A, n.factorial * (n + 1)⟩ := by
  ext
  · rfl
  · simp [permNaiveC, Fintype.card_perm]; ring

theorem detNaiveC_eq {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    detNaiveC A = ⟨detNaive A, n.factorial * (n + 2)⟩ := by
  ext
  · rfl
  · simp [detNaiveC, Fintype.card_perm]; ring

/-! ## Direct Ryser and BB/FG -/

theorem ryserC_eq {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    ryserC A = ⟨ryser A, n ^ 2 * 2 ^ (n - 1) + (n + 2) * 2 ^ n⟩ := by
  ext
  · rfl
  · simp only [ryserC, cost_bigSum, cost_bind, cost_bigProd, cost_pure, cost_tick,
      sum_const_zero, zero_add, sum_const, card_univ, Fintype.card_fin, smul_eq_mul,
      Fintype.card_finset]
    rw [sum_add_distrib, sum_add_distrib, ← mul_sum, sum_card_finset, sum_const, sum_const,
      card_univ, Fintype.card_finset, Fintype.card_fin, smul_eq_mul, smul_eq_mul]
    ring

theorem glynnC_eq {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynnC A = ⟨glynn A, 2 ^ n * ((n + 1) * (2 * n + 2) + 2)⟩ := by
  ext
  · rfl
  · simp [glynnC, Fintype.card_finset]; ring

/-! ## Binary Gray code -/

theorem ryserGrayC_eq {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    ryserGrayC A = ⟨ryserGray A, (2 ^ n - 1) * (2 * n + 3) + n + 1⟩ := by
  have hstep : ∀ k s, (ryserGrayStepC A k s).val = ryserGrayStep A k s := by
    intro k s
    simp only [ryserGrayStepC, ryserGrayStep, val_bind, val_ite, val_tick, val_bigProd,
      val_pure]
    split <;> rfl
  ext
  · simp only [ryserGrayC, val_bind, val_tick, val_loop, val_bigProd, val_pure]
    rw [show (fun k s => (ryserGrayStepC A k s).val) = ryserGrayStep A from
      funext fun k => funext fun s => hstep k s]
    rfl
  · simp only [ryserGrayC, cost_bind, cost_tick, cost_bigProd, cost_pure, sum_const_zero,
      zero_add, card_univ, Fintype.card_fin]
    rw [cost_loop _ _ _ (2 * n + 3)]
    · ring
    · intro k s
      simp [ryserGrayStepC]; ring

theorem glynnGrayC_eq {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynnGrayC A = ⟨glynnGray A, (2 ^ n - 1) * (3 * n + 6) + (n + 1) ^ 2 + (n + 1)⟩ := by
  have hstep : ∀ k s, (glynnGrayStepC A k s).val = glynnGrayStep A k s := by
    intro k s
    simp only [glynnGrayStepC, glynnGrayStep, val_bind, val_ite, val_tick, val_bigProd,
      val_pure]
    split <;> rfl
  ext
  · simp only [glynnGrayC, val_bind, val_loop, val_bigProd, val_pure, val_vec, val_bigSum]
    rw [show (fun k s => (glynnGrayStepC A k s).val) = glynnGrayStep A from
      funext fun k => funext fun s => hstep k s]
    rfl
  · simp only [glynnGrayC, cost_bind, cost_bigProd, cost_pure, cost_vec, cost_bigSum,
      sum_const_zero, zero_add, card_univ, Fintype.card_fin, add_zero, sum_const, smul_eq_mul]
    rw [cost_loop _ _ _ (3 * n + 6)]
    · ring
    · intro k s
      simp [glynnGrayStepC]; ring

theorem glynnWikiC_eq {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynnWikiC A = ⟨glynnWiki A, (2 ^ n - 1) * (3 * n + 4) + (n + 1) ^ 2 + (n + 1)⟩ := by
  have hstep : ∀ k s, (glynnWikiStepC A k s).val = glynnWikiStep A k s := by
    intro k s
    simp only [glynnWikiStepC, glynnWikiStep, val_bind, val_ite, val_tick, val_bigProd,
      val_pure]
    split_ifs <;> rfl
  ext
  · simp only [glynnWikiC, val_bind, val_loop, val_bigProd, val_pure, val_vec, val_bigSum]
    rw [show (fun k s => (glynnWikiStepC A k s).val) = glynnWikiStep A from
      funext fun k => funext fun s => hstep k s]
    rfl
  · simp only [glynnWikiC, cost_bind, cost_bigProd, cost_pure, cost_vec, cost_bigSum,
      sum_const_zero, zero_add, card_univ, Fintype.card_fin, add_zero, sum_const, smul_eq_mul]
    rw [cost_loop _ _ _ (3 * n + 4)]
    · ring
    · intro k s
      simp [glynnWikiStepC]; ring

/-! ## Shifted Ryser (Nijenhuis–Wilf) and normalized BB/FG -/

theorem ryserShiftC_eq {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    ryserShiftC A = ⟨ryserShift A, 2 ^ n * ((n + 1) * (n + 6) + 2) + (n + 1) * (n * 2 ^ (n - 1))⟩ := by
  ext
  · rfl
  · simp only [ryserShiftC, cost_bigSum, cost_bind, cost_bigProd, cost_pure, cost_tick,
      sum_const_zero, zero_add, sum_const, card_univ, Fintype.card_fin, smul_eq_mul,
      Fintype.card_finset]
    simp only [sum_add_distrib, ← mul_sum, sum_card_finset, sum_const, card_univ,
      Fintype.card_finset, Fintype.card_fin, smul_eq_mul]
    ring

theorem ryserShiftGrayC_eq {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    ryserShiftGrayC A = ⟨ryserShiftGray A, (2 ^ n - 1) * (3 * n + 6) + (n + 1) * (n + 5)⟩ := by
  have hstep : ∀ k s, (ryserShiftGrayStepC A k s).val = ryserShiftGrayStep A k s := by
    intro k s
    simp only [ryserShiftGrayStepC, ryserShiftGrayStep, val_bind, val_ite, val_tick,
      val_bigProd, val_pure]
    split <;> rfl
  ext
  · simp only [ryserShiftGrayC, val_bind, val_tick, val_loop, val_bigProd, val_pure, val_vec,
      val_bigSum]
    rw [show (fun k s => (ryserShiftGrayStepC A k s).val) = ryserShiftGrayStep A from
      funext fun k => funext fun s => hstep k s]
    rfl
  · simp only [ryserShiftGrayC, cost_bind, cost_tick, cost_bigProd, cost_pure, cost_vec,
      cost_bigSum, sum_const_zero, zero_add, card_univ, Fintype.card_fin, add_zero, sum_const,
      smul_eq_mul]
    rw [cost_loop _ _ _ (3 * n + 6)]
    · ring
    · intro k s
      simp [ryserShiftGrayStepC]; ring

section Field

variable {K : Type*} [Field K]

theorem ryserNWC_eq {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K) :
    ryserNWC A =
      ⟨ryserNW A, 2 ^ n * ((n + 1) * (n + 5) + 2) + (n + 1) * (n * 2 ^ (n - 1)) + 2⟩ := by
  ext
  · rfl
  · simp only [ryserNWC, cost_bigSum, cost_bind, cost_bigProd, cost_pure, cost_tick,
      sum_const_zero, zero_add, sum_const, card_univ, Fintype.card_fin, smul_eq_mul,
      Fintype.card_finset]
    simp only [sum_add_distrib, ← mul_sum, sum_card_finset, sum_const, card_univ,
      Fintype.card_finset, Fintype.card_fin, smul_eq_mul]
    ring

theorem glynnNormC_eq {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K) (r : Fin (n + 1) → K) :
    glynnNormC A r = ⟨glynnNorm A r,
      (2 ^ n - 1) * (3 * n + 4) + 2 * (n + 1) ^ 2 + 2 * (n + 1) + 1⟩ := by
  ext
  · simp only [glynnNormC, val_bind, val_tick, val_vec, val_bigProd, val_pure, glynnWikiC_eq]
    rfl
  · simp only [glynnNormC, cost_bind, cost_tick, cost_vec, cost_bigProd, cost_pure,
      glynnWikiC_eq, sum_const_zero, zero_add, sum_const, card_univ, Fintype.card_fin,
      smul_eq_mul]
    ring

end Field

/-! ## Multiplicities -/

theorem ryserRepC_eq {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (M : Fin p → ℕ)
    (N : Fin q → ℕ) :
    ryserRepC A M N = ⟨ryserRep A M N,
      (∏ l, (N l + 1)) * (2 * p * q + ∑ k, M k + p + 3)⟩ := by
  ext
  · rfl
  · simp [ryserRepC, sum_add_distrib, Fintype.card_piFinset]; ring

theorem glynnRepC_eq {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (k0 : Fin p)
    (M : Fin p → ℕ) (N : Fin q → ℕ) :
    glynnRepC A k0 M N = ⟨glynnRep A k0 M N,
      (∏ k, (M k + 1)) * (2 * p * q + 2 * q + ∑ l, N l + 3)⟩ := by
  ext
  · rfl
  · simp [glynnRepC, sum_add_distrib, Fintype.card_piFinset]; ring

theorem ryserRepGrayC_eq {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (M : Fin p → ℕ)
    (N : Fin q → ℕ) :
    ryserRepGrayC A M N = ⟨ryserRepGray A M N,
      (∏ l, (N l + 1) - 1) * (2 * p + ∑ k, M k + 4) + ∑ k, M k + p + 1⟩ := by
  have hstep : ∀ t s, (ryserRepGrayStepC A M N t s).val = ryserRepGrayStep A M N t s := by
    intro t s
    simp only [ryserRepGrayStepC, ryserRepGrayStep, val_bind, val_ite, val_tick,
      val_bigProd]
    split <;> rfl
  ext
  · simp only [ryserRepGrayC, val_bind, val_tick, val_loop, val_bigProd]
    rw [show (fun t s => (ryserRepGrayStepC A M N t s).val) = ryserRepGrayStep A M N from
      funext fun t => funext fun s => hstep t s]
    rfl
  · simp only [ryserRepGrayC, cost_bind, cost_tick, cost_bigProd, card_univ, Fintype.card_fin]
    rw [cost_loop _ _ _ (2 * p + ∑ k, M k + 4)]
    · ring
    · intro t s
      simp [ryserRepGrayStepC]; ring

theorem glynnRepGrayC_eq {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (k0 : Fin p)
    (M : Fin p → ℕ) (N : Fin q → ℕ) :
    glynnRepGrayC A k0 M N = ⟨glynnRepGray A k0 M N,
      (∏ k, (M k + 1) - 1) * (3 * q + ∑ l, N l + 4) + q * (2 * p + 1) + ∑ l, N l + q⟩ := by
  have hstep : ∀ t s, (glynnRepGrayStepC A M N t s).val = glynnRepGrayStep A M N t s := by
    intro t s
    simp only [glynnRepGrayStepC, glynnRepGrayStep, val_bind, val_ite, val_tick,
      val_bigProd]
    split <;> rfl
  ext
  · simp only [glynnRepGrayC, val_bind, val_tick, val_loop, val_bigProd, val_pure, val_vec,
      val_bigSum]
    rw [show (fun t s => (glynnRepGrayStepC A M N t s).val) = glynnRepGrayStep A M N from
      funext fun t => funext fun s => hstep t s]
    rfl
  · simp only [glynnRepGrayC, cost_bind, cost_tick, cost_bigProd, cost_pure, cost_vec,
      cost_bigSum, card_univ, Fintype.card_fin, sum_const, smul_eq_mul, add_zero]
    rw [cost_loop _ _ _ (3 * q + ∑ l, N l + 4)]
    · ring
    · intro t s
      simp [glynnRepGrayStepC]; ring

end MaxelerPermanent
