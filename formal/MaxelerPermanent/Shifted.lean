module

public import MaxelerPermanent.Wiki
public import Mathlib.Tactic.FieldSimp

/-!
# Shifted Ryser (Nijenhuis–Wilf) and normalized BB/FG

* `ryserShift_eq`: Ryser with row sums shifted by half the row total, scaled by `2`
  (`ryserShift A = 2ⁿ · perm A`). With the substitution `S ↦ Sᶜ`, the factor
  `2 a_{i,last} - ∑_j a_{ij} + 2 ∑_{j ∈ S} a_{ij}` is the BB/FG column sum of `Aᵀ`
  with the last sign fixed, so this is BB/FG for the transpose (`glynnLast_eq`).
* `ryserShiftGray_eq`: the same in binary Gray-code order (the traditional
  Nijenhuis–Wilf loop).
* `ryserNW_eq`: the published form with halves, over a field with `2 ≠ 0`.
* `glynnNorm_eq`: dividing row `i` by `r i ≠ 0`, running the BB/FG loop and
  multiplying back by `∏ r i` gives `2ⁿ · perm A` (`permanent_scale_rows`).
-/

public section

open Finset

namespace MaxelerPermanent

variable {R : Type*} [CommRing R]

/-- The shifted row sum is the BB/FG sum with signs `+1` on `S ∪ {last}`. -/
theorem ryserShift_factor {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R)
    (S : Finset (Fin n)) (i : Fin (n + 1)) :
    2 * A i (Fin.last n) - ∑ j, A i j + 2 * ∑ j ∈ S, A i j.castSucc =
      A i (Fin.last n) + ∑ j : Fin n, flipSign Sᶜ j * A i j.castSucc := by
  have hterm : ∀ j : Fin n, flipSign Sᶜ j * A i j.castSucc =
      2 * (if j ∈ S then A i j.castSucc else 0) - A i j.castSucc := by
    intro j
    by_cases h : j ∈ S <;> simp [flipSign, h]; ring
  rw [sum_congr rfl fun j _ => hterm j, sum_sub_distrib, ← mul_sum, Fintype.sum_ite_mem,
    Fin.sum_univ_castSucc]
  ring

theorem ryserShift_eq {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    ryserShift A = 2 ^ n * A.permanent := by
  rw [← Matrix.permanent_transpose, ← glynnLast_eq, ryserShift]
  refine Fintype.sum_bijective compl compl_involutive.bijective _ _ fun S => ?_
  rw [card_compl, Fintype.card_fin]
  congr 1
  refine prod_congr rfl fun i _ => ?_
  rw [ryserShift_factor]
  rfl

theorem ryserShiftGray_eq {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    ryserShiftGray A = 2 ^ n * A.permanent := by
  rw [← ryserShift_eq]
  let y : Finset (Fin n) → Fin (n + 1) → R := fun S i =>
    2 * A i (Fin.last n) - ∑ j, A i j + 2 * ∑ j ∈ S, A i j.castSucc
  let term : ℕ → R := fun u =>
    (-1) ^ n * (-1) ^ #(graySet n u) * ∏ i, y (graySet n u) i
  let P : ℕ → GrayState (Fin (n + 1)) R → Prop := fun t s =>
    s.sums = y (graySet n t) ∧ s.sign = (-1) ^ n * (-1) ^ #(graySet n t) ∧
      s.acc = ∑ u ∈ range (t + 1), term u
  have hinit : (fun i => 2 * A i (Fin.last n) - ∑ j, A i j) = y (graySet n 0) := by
    funext i; simp [y, graySet_zero]
  have hP := runSteps_induction (ryserShiftGrayStep A)
    ⟨fun i => 2 * A i (Fin.last n) - ∑ j, A i j, (-1) ^ n,
      (-1) ^ n * ∏ i, (2 * A i (Fin.last n) - ∑ j, A i j)⟩ P (2 ^ n - 1) ?base ?step
    (2 ^ n - 1) le_rfl
  case base =>
    refine ⟨hinit, by simp [graySet_zero], ?_⟩
    simp only [zero_add, range_one, sum_singleton, term, graySet_zero, card_empty, pow_zero,
      mul_one]
    congr 1
    exact prod_congr rfl fun i _ => by simp [y]
  case step =>
    rintro t ⟨sums, sign, acc⟩ ht ⟨hsums, hsign, hacc⟩
    simp only at hsums hsign hacc
    subst hsums hsign hacc
    have ht' : t + 1 < 2 ^ n := by have := Nat.one_le_two_pow (n := n); omega
    obtain ⟨hj, hcase⟩ := graySet_succ ht'
    have hcol : (fun i => natIdx (A i) (trailingZeros (t + 1))) =
        fun i => A i (⟨_, hj⟩ : Fin n).castSucc := by
      funext i; rw [natIdx_of_lt _ (by omega)]; rfl
    simp only [ryserShiftGrayStep, hcol]
    split_ifs at hcase ⊢ with hbit
    · obtain ⟨hnot, hS⟩ := hcase
      have hsums : y (graySet n t) + 2 • (fun i => A i (⟨_, hj⟩ : Fin n).castSucc) =
          y (graySet n (t + 1)) := by
        funext i
        simp only [y, hS, sum_insert hnot, Pi.add_apply, two_nsmul]
        ring
      have hsign : -((-1 : R) ^ n * (-1) ^ #(graySet n t)) =
          (-1) ^ n * (-1) ^ #(graySet n (t + 1)) := by
        rw [hS, card_insert_of_notMem hnot, pow_succ]; ring
      refine ⟨hsums, hsign, ?_⟩
      dsimp only
      rw [sum_range_succ _ (t + 1), hsign, hsums]
    · obtain ⟨hmem, hS⟩ := hcase
      have hsums : y (graySet n t) - 2 • (fun i => A i (⟨_, hj⟩ : Fin n).castSucc) =
          y (graySet n (t + 1)) := by
        funext i
        simp only [y, hS, Pi.sub_apply, Pi.add_apply, two_nsmul]
        rw [← add_sum_erase _ _ hmem]
        ring
      have hsign : -((-1 : R) ^ n * (-1) ^ #(graySet n t)) =
          (-1) ^ n * (-1) ^ #(graySet n (t + 1)) := by
        have hpos : 0 < #(graySet n t) := card_pos.mpr ⟨_, hmem⟩
        rw [hS, card_erase_of_mem hmem]
        conv_lhs => rw [show #(graySet n t) = #(graySet n t) - 1 + 1 by omega, pow_succ]
        ring
      refine ⟨hsums, hsign, ?_⟩
      dsimp only
      rw [sum_range_succ _ (t + 1), hsign, hsums]
  rw [ryserShiftGray, hP.2.2,
    show 2 ^ n - 1 + 1 = 2 ^ n by have := Nat.one_le_two_pow (n := n); omega,
    sum_range_graySet (fun S => (-1 : R) ^ n * (-1) ^ #S * ∏ i, y S i), ryserShift]
  refine sum_congr rfl fun S _ => ?_
  have : #S ≤ n := by simpa using card_le_univ S
  rw [neg_one_pow_sub this]

section Field

variable {K : Type*} [Field K]

theorem ryserNW_eq {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K) (h2 : (2 : K) ≠ 0) :
    ryserNW A = A.permanent := by
  have hfac : ∀ (S : Finset (Fin n)) i,
      A i (Fin.last n) - (∑ j, A i j) / 2 + ∑ j ∈ S, A i j.castSucc =
        (2 * A i (Fin.last n) - ∑ j, A i j + 2 * ∑ j ∈ S, A i j.castSucc) / 2 := by
    intro S i; field_simp
  have hsq : (-1 : K) ^ n * (-1) ^ n = 1 := by rw [← mul_pow]; norm_num
  have key := ryserShift_eq A
  rw [ryserShift] at key
  rw [ryserNW]
  simp_rw [hfac, prod_div_distrib, prod_const, card_univ, Fintype.card_fin]
  have hsum : ∑ S : Finset (Fin n), (-1 : K) ^ #S *
      ((∏ i, (2 * A i (Fin.last n) - ∑ j, A i j + 2 * ∑ j ∈ S, A i j.castSucc)) / 2 ^ (n + 1)) =
      (-1) ^ n * (2 ^ n * A.permanent) / 2 ^ (n + 1) := by
    rw [← key, mul_sum, div_eq_mul_inv, sum_mul]
    refine sum_congr rfl fun S _ => ?_
    have : #S ≤ n := by simpa using card_le_univ S
    rw [neg_one_pow_sub this]
    calc (-1 : K) ^ #S * ((∏ i, (2 * A i (Fin.last n) - ∑ j, A i j +
          2 * ∑ j ∈ S, A i j.castSucc)) / 2 ^ (n + 1))
        = ((-1) ^ n * (-1) ^ n) * (-1) ^ #S * ((∏ i, (2 * A i (Fin.last n) - ∑ j, A i j +
          2 * ∑ j ∈ S, A i j.castSucc)) / 2 ^ (n + 1)) := by rw [hsq, one_mul]
      _ = _ := by ring
  rw [hsum]
  have h : (2 : K) * 2 ^ n / 2 ^ (n + 1) = 1 := by rw [← pow_succ', div_self (pow_ne_zero _ h2)]
  calc _ = ((-1 : K) ^ n * (-1) ^ n) * (2 * 2 ^ n / 2 ^ (n + 1)) * A.permanent := by ring
    _ = A.permanent := by rw [hsq, h]; ring

end Field

/-- Scaling row `i` by `c i` scales the permanent by `∏ c i`. -/
theorem permanent_scale_rows {m : ℕ} (c : Fin m → R) (A : Matrix (Fin m) (Fin m) R) :
    (Matrix.of fun i j => c i * A i j).permanent = (∏ i, c i) * A.permanent := by
  simp only [Matrix.permanent, Matrix.of_apply, prod_mul_distrib, Equiv.prod_comp, mul_sum]

section Field

variable {K : Type*} [Field K]

theorem glynnNorm_eq {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K)
    (r : Fin (n + 1) → K) (hr : ∀ i, r i ≠ 0) :
    glynnNorm A r = 2 ^ n * A.permanent := by
  have hB : (Matrix.of fun i j => A i j / r i) = Matrix.of fun i j => (r i)⁻¹ * A i j := by
    ext i j; simp [div_eq_inv_mul]
  have hprod : ∏ i, r i ≠ 0 := prod_ne_zero_iff.mpr fun i _ => hr i
  rw [glynnNorm, glynnWiki_eq, hB, permanent_scale_rows, prod_inv_distrib]
  field_simp

end Field

end MaxelerPermanent
