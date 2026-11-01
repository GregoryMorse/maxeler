module

public import MaxelerPermanent.GrayAlgorithms
public import Mathlib.Logic.Equiv.Fin.Rotate

/-!
# BB/FG in the form of `permanent_benchmark.py`

`glynnWiki` is the loop of `permanent_glynn_gray_exact`, the Gray-code form of
Glynn's formula listed on Wikipedia:

* the **last** row has fixed sign `+1`;
* `δ` is toggled in place at `idx = tz i`;
* the sign of the `i`-th term is `(-1)^i`, not tracked from `δ`.

`glynnWiki_eq` proves `glynnWiki A = 2ⁿ · perm A`. The proof shows:

* the toggled `δ` is the binary Gray code `graySet`;
* `(-1)^i = (-1)^|S_i|`, because each step changes the subset by one element;
* moving the last row to the front (`finRotate`) does not change the permanent.
-/

public section

open Finset

namespace MaxelerPermanent

variable {R : Type*} [CommRing R]

/-- The size of the Gray-code subset at step `t` has the parity of `t`. -/
theorem neg_one_pow_card_graySet {n t : ℕ} (ht : t < 2 ^ n) :
    (-1 : R) ^ #(graySet n t) = (-1) ^ t := by
  induction t with
  | zero => simp [graySet_zero]
  | succ t ih =>
    obtain ⟨hj, hcase⟩ := graySet_succ ht
    rw [pow_succ, ← ih (by omega)]
    split_ifs at hcase
    · rw [hcase.2, card_insert_of_notMem hcase.1, pow_succ]
    · have hpos : 0 < #(graySet n t) := card_pos.mpr ⟨_, hcase.1⟩
      rw [hcase.2, card_erase_of_mem hcase.1]
      conv_rhs => rw [show #(graySet n t) = #(graySet n t) - 1 + 1 by omega, pow_succ]
      rw [mul_assoc]; simp

theorem glynnWiki_eq {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynnWiki A = 2 ^ n * A.permanent := by
  -- column sums with the last row fixed
  let cl : Finset (Fin n) → Fin (n + 1) → R := fun S j =>
    A (Fin.last n) j + ∑ i : Fin n, flipSign S i * A i.castSucc j
  let term : ℕ → R := fun u => (-1) ^ u * ∏ j, cl (graySet n u) j
  let P : ℕ → WikiState n R → Prop := fun t s =>
    s.delta = (fun k => decide (k ∉ graySet n t)) ∧ s.sums = cl (graySet n t) ∧
      s.acc = ∑ u ∈ range (t + 1), term u
  have hinit : (fun j => ∑ i, A i j) = cl (graySet n 0) := by
    funext j
    simp [cl, graySet_zero, flipSign, Fin.sum_univ_castSucc, add_comm]
  have hP := runSteps_induction (glynnWikiStep A)
    ⟨fun _ => true, fun j => ∑ i, A i j, ∏ j, ∑ i, A i j⟩ P (2 ^ n - 1) ?base ?step
    (2 ^ n - 1) le_rfl
  case base =>
    refine ⟨by funext k; simp [graySet_zero], hinit, ?_⟩
    simp only [zero_add, range_one, sum_singleton, term, pow_zero, one_mul]
    exact prod_congr rfl fun j _ => congrFun hinit j
  case step =>
    rintro t ⟨delta, sums, acc⟩ ht ⟨hdelta, hsums, hacc⟩
    simp only at hdelta hsums hacc
    subst hdelta hsums hacc
    have ht' : t + 1 < 2 ^ n := by have := Nat.one_le_two_pow (n := n); omega
    obtain ⟨hj, hcase⟩ := graySet_succ ht'
    set a : Fin n := ⟨trailingZeros (t + 1), hj⟩ with ha
    have hrow : natIdx A (trailingZeros (t + 1)) = A a.castSucc := by
      unfold natIdx; rw [dif_pos (by omega)]; rfl
    -- the toggled sign vector is the next Gray code
    have hdel : toggleAt (fun k => decide (k ∉ graySet n t)) (trailingZeros (t + 1)) =
        fun k => decide (k ∉ graySet n (t + 1)) := by
      funext k
      simp only [toggleAt]
      by_cases hk : k.val = trailingZeros (t + 1)
      · have hka : k = a := Fin.ext hk
        subst hka
        rw [if_pos rfl]
        split_ifs at hcase
        · rw [hcase.2]; simp [hcase.1]
        · rw [hcase.2]; simp [hcase.1]
      · have hka : k ≠ a := fun h => hk (congrArg Fin.val h)
        rw [if_neg hk]
        split_ifs at hcase
        · rw [hcase.2]; simp [hka]
        · rw [hcase.2]; simp [hka]
    have hbit : (if h : trailingZeros (t + 1) < n then
        (fun k => decide (k ∉ graySet n (t + 1))) ⟨_, h⟩ else false) =
        decide (a ∉ graySet n (t + 1)) := by
      rw [dif_pos hj]
    have hsums' : (if decide (a ∉ graySet n (t + 1)) then
        cl (graySet n t) + 2 • A a.castSucc else cl (graySet n t) - 2 • A a.castSucc) =
        cl (graySet n (t + 1)) := by
      split_ifs at hcase with hb
      · obtain ⟨hnot, hS⟩ := hcase
        rw [if_neg (by simp [hS])]
        funext j
        simp only [cl, hS, Pi.sub_apply, flipSign_insert hnot, sub_mul, sum_sub_distrib,
          ite_mul, zero_mul, sum_ite_eq', mem_univ, if_true, two_nsmul, Pi.add_apply]
        ring
      · obtain ⟨hmem, hS⟩ := hcase
        rw [if_pos (by simp [hS])]
        funext j
        simp only [cl, hS, Pi.add_apply, flipSign_erase hmem, add_mul, sum_add_distrib,
          ite_mul, zero_mul, sum_ite_eq', mem_univ, if_true, two_nsmul]
        ring
    simp only [glynnWikiStep, hdel, hbit, hrow, hsums']
    refine ⟨rfl, rfl, ?_⟩
    rw [sum_range_succ _ (t + 1)]
    simp only [term]
    rcases Nat.mod_two_eq_zero_or_one (t + 1) with h | h
    · rw [if_neg (by omega), Even.neg_one_pow (Nat.even_iff.mpr h), one_mul]
    · rw [if_pos h, Odd.neg_one_pow (Nat.odd_iff.mpr h)]; ring
  rw [glynnWiki, hP.2.2, show 2 ^ n - 1 + 1 = 2 ^ n by have := Nat.one_le_two_pow (n := n); omega]
  have hsign : ∑ u ∈ range (2 ^ n), term u =
      ∑ u ∈ range (2 ^ n), (-1 : R) ^ #(graySet n u) * ∏ j, cl (graySet n u) j :=
    sum_congr rfl fun u hu => by rw [neg_one_pow_card_graySet (mem_range.mp hu)]
  rw [hsign, sum_range_graySet (fun S => (-1 : R) ^ #S * ∏ j, cl S j)]
  -- move the last row to the front
  let e := finRotate (n + 1)
  let B := A.submatrix e.symm id
  have hB0 : ∀ j, B 0 j = A (Fin.last n) j := by
    intro j
    simp only [B, Matrix.submatrix_apply, id]
    rw [show e.symm 0 = Fin.last n from (Equiv.symm_apply_eq e).mpr finRotate_last.symm]
  have hBs : ∀ (i : Fin n) j, B i.succ j = A i.castSucc j := by
    intro i j
    simp only [B, Matrix.submatrix_apply, id]
    congr 1
    rw [Equiv.symm_apply_eq]
    ext
    rw [coe_finRotate_of_ne_last (Fin.castSucc_lt_last i).ne]
    simp
  have hperm : B.permanent = A.permanent := Matrix.permanent_permute_cols e.symm A
  rw [← hperm, ← glynn_eq, glynn]
  refine sum_congr rfl fun S _ => ?_
  congr 1
  refine prod_congr rfl fun j _ => ?_
  simp only [cl, hB0, hBs]

end MaxelerPermanent
