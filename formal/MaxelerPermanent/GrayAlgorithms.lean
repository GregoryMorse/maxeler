module

public import MaxelerPermanent.Repeated
public import MaxelerPermanent.GrayCode
public import Mathlib.Tactic.LinearCombination

/-!
# Gray-code algorithms

Correctness of the loop-based algorithms:

* `ryserGray_eq_permanent` — Ryser in binary Gray-code order (Nijenhuis–Wilf);
* `glynnGray_eq` — BB/FG in binary Gray-code order;
* `ryserRepGray_eq_ryserRep` — Ryser with multiplicities in n-ary Gray-code order;
* `glynnRepGray_eq_glynnRep` — BB/FG with multiplicities in n-ary Gray-code order,
  with the binomial weight updated by exact integer division.

Each proof is a loop invariant: after step `t` the running sums, sign and binomial
weight are those of the `t`-th Gray code, and the accumulator is the partial sum of
the closed formula over the first `t + 1` codes. The Gray-code enumeration is a
bijection onto all subsets / occupation vectors (`sum_range_graySet`,
`sum_range_naryGray`), which turns the final accumulator into the closed formula.
-/

public section

open Finset

namespace MaxelerPermanent

variable {R : Type*} [CommRing R]

/-- Loop invariants are established by induction on the step counter. -/
theorem runSteps_induction {S : Type*} (step : ℕ → S → S) (init : S) (P : ℕ → S → Prop)
    (T : ℕ) (h0 : P 0 init) (hs : ∀ t s, t + 1 ≤ T → P t s → P (t + 1) (step (t + 1) s)) :
    ∀ t ≤ T, P t (runSteps step init t) := by
  intro t ht
  induction t with
  | zero => exact h0
  | succ t ih => exact hs t _ ht (ih (by omega))

theorem natIdx_of_lt {m : ℕ} {α : Type*} [Zero α] (f : Fin m → α) {i : ℕ} (h : i < m) :
    natIdx f i = f ⟨i, h⟩ := by
  simp [natIdx, h]

theorem flipSign_insert {ι : Type*} [DecidableEq ι] {S : Finset ι} {a : ι} (ha : a ∉ S)
    (i : ι) : (flipSign (insert a S) i : R) = flipSign S i - if i = a then 2 else 0 := by
  by_cases hi : i = a
  · subst hi; simp [flipSign, ha]; ring
  · simp [flipSign, hi]

theorem flipSign_erase {ι : Type*} [DecidableEq ι] {S : Finset ι} {a : ι} (ha : a ∈ S)
    (i : ι) : (flipSign (S.erase a) i : R) = flipSign S i + if i = a then 2 else 0 := by
  by_cases hi : i = a
  · subst hi; simp [flipSign, ha]; ring
  · simp [flipSign, hi]

/-! ## Binary Gray code: Ryser -/

theorem ryserGray_eq_permanent {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    ryserGray A = A.permanent := by
  rw [← ryser_eq_permanent]
  let term : ℕ → R := fun u =>
    (-1) ^ #(graySet n u) * ∏ i, ∑ j ∈ graySet n u, A i j
  let P : ℕ → GrayState (Fin n) R → Prop := fun t s =>
    s.sums = (fun i => ∑ j ∈ graySet n t, A i j) ∧ s.sign = (-1) ^ #(graySet n t) ∧
      s.acc = ∑ u ∈ range (t + 1), term u
  have hP := runSteps_induction (ryserGrayStep A) ⟨0, 1, ∏ _i : Fin n, (0 : R)⟩ P
    (2 ^ n - 1) ?base ?step (2 ^ n - 1) le_rfl
  case base =>
    refine ⟨?_, ?_, ?_⟩ <;> simp [term, graySet_zero]
    rfl
  case step =>
    rintro t ⟨sums, sign, acc⟩ ht ⟨hsums, hsign, hacc⟩
    simp only at hsums hsign hacc
    subst hsums hsign hacc
    have ht' : t + 1 < 2 ^ n := by have := Nat.one_le_two_pow (n := n); omega
    obtain ⟨hj, hcase⟩ := graySet_succ ht'
    simp only [ryserGrayStep, natIdx_of_lt _ hj]
    split_ifs at hcase ⊢ with hbit
    · obtain ⟨hnot, hS⟩ := hcase
      have hsums : (fun i => ∑ j ∈ graySet n t, A i j) + (fun i => A i ⟨_, hj⟩) =
          fun i => ∑ j ∈ graySet n (t + 1), A i j := by
        funext i; simp [hS, sum_insert hnot, add_comm]
      have hsign : -(-1 : R) ^ #(graySet n t) = (-1) ^ #(graySet n (t + 1)) := by
        rw [hS, card_insert_of_notMem hnot, pow_succ]; ring
      refine ⟨hsums, hsign, ?_⟩
      dsimp only
      rw [sum_range_succ _ (t + 1), hsign]
      congr 1
      simp only [term]
      congr 1
      exact prod_congr rfl fun i _ => congrFun hsums i
    · obtain ⟨hmem, hS⟩ := hcase
      have hsums : (fun i => ∑ j ∈ graySet n t, A i j) - (fun i => A i ⟨_, hj⟩) =
          fun i => ∑ j ∈ graySet n (t + 1), A i j := by
        funext i; simp [hS, sum_erase_eq_sub hmem]
      have hsign : -(-1 : R) ^ #(graySet n t) = (-1) ^ #(graySet n (t + 1)) := by
        have hpos : 0 < #(graySet n t) := card_pos.mpr ⟨_, hmem⟩
        rw [hS, card_erase_of_mem hmem]
        conv_lhs => rw [show #(graySet n t) = #(graySet n t) - 1 + 1 by omega, pow_succ]
        ring
      refine ⟨hsums, hsign, ?_⟩
      dsimp only
      rw [sum_range_succ _ (t + 1), hsign]
      congr 1
      simp only [term]
      congr 1
      exact prod_congr rfl fun i _ => congrFun hsums i
  rw [ryserGray, hP.2.2, show 2 ^ n - 1 + 1 = 2 ^ n by have := Nat.one_le_two_pow (n := n); omega,
    sum_range_graySet (fun S => (-1 : R) ^ #S * ∏ i, ∑ j ∈ S, A i j), ryser, mul_sum]
  refine sum_congr rfl fun S _ => ?_
  have : #S ≤ n := by simpa using card_le_univ S
  rw [neg_one_pow_sub this]
  ring

/-! ## Binary Gray code: BB/FG -/

theorem glynnGray_eq {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynnGray A = 2 ^ n * A.permanent := by
  rw [← glynn_eq]
  let colsum : Finset (Fin n) → Fin (n + 1) → R := fun S j =>
    A 0 j + ∑ i : Fin n, flipSign S i * A i.succ j
  let term : ℕ → R := fun u => (-1) ^ #(graySet n u) * ∏ j, colsum (graySet n u) j
  let P : ℕ → GrayState (Fin (n + 1)) R → Prop := fun t s =>
    s.sums = colsum (graySet n t) ∧ s.sign = (-1) ^ #(graySet n t) ∧
      s.acc = ∑ u ∈ range (t + 1), term u
  have hinit : (fun j => ∑ i, A i j) = colsum (graySet n 0) := by
    funext j
    simp [colsum, graySet_zero, flipSign, Fin.sum_univ_succ]
  have hP := runSteps_induction (glynnGrayStep A)
    ⟨fun j => ∑ i, A i j, 1, ∏ j, ∑ i, A i j⟩ P (2 ^ n - 1) ?base ?step (2 ^ n - 1) le_rfl
  case base =>
    refine ⟨hinit, by simp [graySet_zero], ?_⟩
    rw [graySet_zero] at hinit
    simp only [zero_add, range_one, sum_singleton, term, graySet_zero, card_empty, pow_zero,
      one_mul]
    exact prod_congr rfl fun j _ => congrFun hinit j
  case step =>
    rintro t ⟨sums, sign, acc⟩ ht ⟨hsums, hsign, hacc⟩
    simp only at hsums hsign hacc
    subst hsums hsign hacc
    have ht' : t + 1 < 2 ^ n := by have := Nat.one_le_two_pow (n := n); omega
    obtain ⟨hj, hcase⟩ := graySet_succ ht'
    have hj1 : trailingZeros (t + 1) + 1 < n + 1 := by omega
    have hrow : natIdx A (trailingZeros (t + 1) + 1) = A (Fin.succ ⟨_, hj⟩) := by
      unfold natIdx; rw [dif_pos hj1]; rfl
    simp only [glynnGrayStep, hrow]
    split_ifs at hcase ⊢ with hbit
    · obtain ⟨hnot, hS⟩ := hcase
      have hsums : colsum (graySet n t) - 2 • A (Fin.succ ⟨_, hj⟩) =
          colsum (graySet n (t + 1)) := by
        funext j
        simp only [colsum, hS, Pi.sub_apply, Pi.smul_apply, flipSign_insert hnot, sub_mul,
          sum_sub_distrib, ite_mul, zero_mul, sum_ite_eq', mem_univ, if_true, two_nsmul, Pi.add_apply]
        push_cast; ring
      have hsign : -(-1 : R) ^ #(graySet n t) = (-1) ^ #(graySet n (t + 1)) := by
        rw [hS, card_insert_of_notMem hnot, pow_succ]; ring
      refine ⟨hsums, hsign, ?_⟩
      dsimp only
      rw [sum_range_succ _ (t + 1), hsign]
      congr 1
      simp only [term]
      congr 1
      exact prod_congr rfl fun i _ => congrFun hsums i
    · obtain ⟨hmem, hS⟩ := hcase
      have hsums : colsum (graySet n t) + 2 • A (Fin.succ ⟨_, hj⟩) =
          colsum (graySet n (t + 1)) := by
        funext j
        simp only [colsum, hS, Pi.add_apply, Pi.smul_apply, flipSign_erase hmem, add_mul,
          sum_add_distrib, ite_mul, zero_mul, sum_ite_eq', mem_univ, if_true, two_nsmul, Pi.add_apply]
        push_cast; ring
      have hsign : -(-1 : R) ^ #(graySet n t) = (-1) ^ #(graySet n (t + 1)) := by
        have hpos : 0 < #(graySet n t) := card_pos.mpr ⟨_, hmem⟩
        rw [hS, card_erase_of_mem hmem]
        conv_lhs => rw [show #(graySet n t) = #(graySet n t) - 1 + 1 by omega, pow_succ]
        ring
      refine ⟨hsums, hsign, ?_⟩
      dsimp only
      rw [sum_range_succ _ (t + 1), hsign]
      congr 1
      simp only [term]
      congr 1
      exact prod_congr rfl fun i _ => congrFun hsums i
  rw [glynnGray, hP.2.2, show 2 ^ n - 1 + 1 = 2 ^ n by have := Nat.one_le_two_pow (n := n); omega,
    sum_range_graySet (fun S => (-1 : R) ^ #S * ∏ j, colsum S j), glynn]

/-! ## n-ary Gray code -/

theorem naryGray_zero {m : ℕ} (rad : Fin m → ℕ) : naryGray rad 0 = 0 := by
  induction m with
  | zero => funext i; exact i.elim0
  | succ m ih =>
    funext i
    refine Fin.cases ?_ (fun i => ?_) i
    · simp [naryGray_zero_digit]
    · rw [naryGray_succ_digit, Nat.zero_div, ih]; rfl

/-- Changing one coordinate of `g` changes a sum `∑_k F k (g k)` by the difference
of the two terms at that coordinate. -/
theorem sum_of_eq_off {m : ℕ} {M : Type*} [AddCommMonoid M] {g g' : Fin m → ℕ} {i : Fin m}
    (h : ∀ k, k ≠ i → g' k = g k) (F : Fin m → ℕ → M) :
    ∑ k, F k (g' k) + F i (g i) = ∑ k, F k (g k) + F i (g' i) := by
  rw [Fintype.sum_eq_add_sum_compl i, Fintype.sum_eq_add_sum_compl i (fun k => F k (g k))]
  have : ∑ k ∈ {i}ᶜ, F k (g' k) = ∑ k ∈ {i}ᶜ, F k (g k) :=
    sum_congr rfl fun k hk => by rw [h k (by simpa using hk)]
  rw [this]
  abel

theorem prod_of_eq_off {m : ℕ} {g g' : Fin m → ℕ} {i : Fin m}
    (h : ∀ k, k ≠ i → g' k = g k) (F : Fin m → ℕ → ℕ) :
    ∃ P, ∏ k, F k (g k) = F i (g i) * P ∧ ∏ k, F k (g' k) = F i (g' i) * P := by
  refine ⟨∏ k ∈ {i}ᶜ, F k (g k), Fintype.prod_eq_mul_prod_compl i _, ?_⟩
  rw [Fintype.prod_eq_mul_prod_compl i]
  congr 1
  exact prod_congr rfl fun k hk => by rw [h k (by simpa using hk)]

/-- The exact-division binomial update of the C++ code. -/
theorem binomStep_eq {N u v P : ℕ} (hu : u ≤ N) (hv : v ≤ N) (huv : v + 1 = u ∨ u + 1 = v) :
    binomStep (N.choose u * P) N u v = N.choose v * P := by
  rw [binomStep]
  rcases huv with h | h
  · rw [if_pos (by omega)]
    have key : N.choose u * u = N.choose v * (N - v) := by
      rw [← h]; exact Nat.choose_succ_right_eq N v
    have hpos : 0 < N - v := by omega
    rw [show N.choose u * P * u = N.choose v * P * (N - v) by
      rw [mul_right_comm, key, mul_right_comm]]
    exact Nat.mul_div_cancel _ hpos
  · rw [if_neg (by omega)]
    have key : N.choose v * v = N.choose u * (N - u) := by
      rw [← h]; exact Nat.choose_succ_right_eq N u
    have hpos : 0 < v := by omega
    rw [show N.choose u * P * (N - u) = N.choose v * P * v by
      rw [mul_right_comm, ← key, mul_right_comm]]
    exact Nat.mul_div_cancel _ hpos

theorem neg_one_pow_of_step {a a' u v : ℕ} (h : a' + u = a + v) (huv : v + 1 = u ∨ u + 1 = v) :
    -(-1 : R) ^ a = (-1) ^ a' := by
  rcases huv with h' | h'
  · rw [show a = a' + 1 by omega, pow_succ]; ring
  · rw [show a' = a + 1 by omega, pow_succ]; ring

/-- Data of one n-ary Gray-code step, in the form the loop bodies consume. -/
theorem naryGray_step_data {m : ℕ} (rad : Fin m → ℕ) {t : ℕ} (ht : t + 1 < ∏ k, rad k) :
    ∃ h : naryGrayIdx rad t < m,
      (∀ k, k ≠ ⟨_, h⟩ → naryGray rad (t + 1) k = naryGray rad t k) ∧
      (natIdx (naryGray rad (t + 1)) (naryGrayIdx rad t) + 1 =
          natIdx (naryGray rad t) (naryGrayIdx rad t) ∨
        natIdx (naryGray rad t) (naryGrayIdx rad t) + 1 =
          natIdx (naryGray rad (t + 1)) (naryGrayIdx rad t)) ∧
      natIdx (naryGray rad t) (naryGrayIdx rad t) = naryGray rad t ⟨_, h⟩ ∧
      natIdx (naryGray rad (t + 1)) (naryGrayIdx rad t) = naryGray rad (t + 1) ⟨_, h⟩ := by
  obtain ⟨h, hoth, hstep⟩ := naryGray_succ rad ht
  refine ⟨h, hoth, ?_, natIdx_of_lt _ h, natIdx_of_lt _ h⟩
  rwa [natIdx_of_lt _ h, natIdx_of_lt _ h]

theorem one_le_prod_succ {m : ℕ} (N : Fin m → ℕ) : 1 ≤ ∏ k, (N k + 1) :=
  Nat.one_le_iff_ne_zero.mpr (prod_ne_zero_iff.mpr fun k _ => by omega)

/-- Row/column sums linear in the digits change by `± w i` in one Gray-code step. -/
theorem lin_step {m : ℕ} {ι : Type*} {g g' : Fin m → ℕ} {i : Fin m}
    (hoth : ∀ k, k ≠ i → g' k = g k) (hstep : g' i + 1 = g i ∨ g i + 1 = g' i)
    (w : Fin m → ι → R) :
    (if g i < g' i then (fun x => ∑ l, (g l : R) * w l x) + w i
      else (fun x => ∑ l, (g l : R) * w l x) - w i) = fun x => ∑ l, (g' l : R) * w l x := by
  funext x
  have h := sum_of_eq_off (M := R) hoth (fun l y => (y : R) * w l x)
  split_ifs with hlt
  · have e : (g' i : R) = g i + 1 := by
      rw [show g' i = g i + 1 by omega]; push_cast; ring
    rw [e] at h
    simp only [Pi.add_apply]
    linear_combination h.symm
  · have e : (g i : R) = g' i + 1 := by
      rw [show g i = g' i + 1 by omega]; push_cast; ring
    rw [e] at h
    simp only [Pi.sub_apply]
    linear_combination h.symm

/-- The BB/FG column sums change by `∓ 2 w i` in one Gray-code step. -/
theorem cs_step {m : ℕ} {ι : Type*} {g g' : Fin m → ℕ} {i : Fin m}
    (hoth : ∀ k, k ≠ i → g' k = g k) (hstep : g' i + 1 = g i ∨ g i + 1 = g' i)
    (base : ι → R) (Mv : Fin m → ℕ) (w : Fin m → ι → R) :
    (if g i < g' i then (fun x => base x + ∑ l, ((Mv l : R) - 2 * (g l : R)) * w l x) - 2 • w i
      else (fun x => base x + ∑ l, ((Mv l : R) - 2 * (g l : R)) * w l x) + 2 • w i) =
      fun x => base x + ∑ l, ((Mv l : R) - 2 * (g' l : R)) * w l x := by
  funext x
  have h := sum_of_eq_off (M := R) hoth (fun l y => ((Mv l : R) - 2 * (y : R)) * w l x)
  split_ifs with hlt
  · have e : (g' i : R) = g i + 1 := by
      rw [show g' i = g i + 1 by omega]; push_cast; ring
    rw [e] at h
    simp only [Pi.sub_apply, two_nsmul, Pi.add_apply]
    linear_combination h.symm
  · have e : (g i : R) = g' i + 1 := by
      rw [show g i = g' i + 1 by omega]; push_cast; ring
    rw [e] at h
    simp only [Pi.add_apply, two_nsmul]
    linear_combination h.symm

/-- The binomial weight after one Gray-code step. -/
theorem binom_step {m : ℕ} {g g' : Fin m → ℕ} {i : Fin m} (Nv : Fin m → ℕ)
    (hoth : ∀ k, k ≠ i → g' k = g k) (hstep : g' i + 1 = g i ∨ g i + 1 = g' i)
    (hu : g i ≤ Nv i) (hv : g' i ≤ Nv i) :
    binomStep (∏ l, (Nv l).choose (g l)) (Nv i) (g i) (g' i) = ∏ l, (Nv l).choose (g' l) := by
  obtain ⟨Pb, hb, hb'⟩ := prod_of_eq_off (i := i) hoth (fun l x => (Nv l).choose x)
  rw [hb, hb']
  exact binomStep_eq hu hv hstep

/-! ## n-ary Gray code: Ryser with multiplicities -/

theorem ryserRepGray_eq_ryserRep {p q : ℕ} (A : Matrix (Fin p) (Fin q) R)
    (M : Fin p → ℕ) (N : Fin q → ℕ) : ryserRepGray A M N = ryserRep A M N := by
  let rs : (Fin q → ℕ) → Fin p → R := fun s k => ∑ l, (s l : R) * A k l
  let term : (Fin q → ℕ) → R := fun s =>
    (-1) ^ (∑ l, s l) * ((∏ l, (N l).choose (s l) : ℕ) : R) * ∏ k, rs s k ^ M k
  let P : ℕ → RepState (Fin p) R → Prop := fun t st =>
    st.sums = rs (naryGray (fun l => N l + 1) t) ∧
      st.sign = (-1) ^ (∑ l, naryGray (fun l => N l + 1) t l) ∧
      st.binom = ∏ l, (N l).choose (naryGray (fun l => N l + 1) t l) ∧
      st.acc = ∑ u ∈ range (t + 1), term (naryGray (fun l => N l + 1) u)
  have hK := one_le_prod_succ N
  have hP := runSteps_induction (ryserRepGrayStep A M N) ⟨0, 1, 1, ∏ k, (0 : R) ^ M k⟩ P
    (∏ l, (N l + 1) - 1) ?base ?step _ le_rfl
  case base =>
    refine ⟨?_, ?_, ?_, ?_⟩
    · funext k; simp [rs, naryGray_zero]
    · simp [naryGray_zero]
    · simp [naryGray_zero]
    · simp [term, rs, naryGray_zero]
  case step =>
    rintro t ⟨sums, sign, binom, acc⟩ ht ⟨hsums, hsign, hbinom, hacc⟩
    simp only at hsums hsign hbinom hacc
    subst hsums hsign hbinom hacc
    have ht' : t + 1 < ∏ l, (N l + 1) := by omega
    obtain ⟨hi, hoth, hstep, hu, hv⟩ := naryGray_step_data (fun l => N l + 1) ht'
    rw [hu, hv] at hstep
    have hul : naryGray (fun l => N l + 1) t ⟨_, hi⟩ ≤ N ⟨_, hi⟩ :=
      Nat.lt_succ_iff.mp (naryGray_lt _ (by omega) _)
    have hvl : naryGray (fun l => N l + 1) (t + 1) ⟨_, hi⟩ ≤ N ⟨_, hi⟩ :=
      Nat.lt_succ_iff.mp (naryGray_lt _ ht' _)
    have hcol : (fun k => natIdx (A k) (naryGrayIdx (fun l => N l + 1) t)) =
        (fun l k => A k l) ⟨_, hi⟩ := by
      funext k; exact natIdx_of_lt _ hi
    have hsums := lin_step (R := R) hoth hstep (fun l k => A k l)
    have hbin := binom_step N hoth hstep hul hvl
    have hsgn : -(-1 : R) ^ (∑ l, naryGray (fun l => N l + 1) t l) =
        (-1) ^ (∑ l, naryGray (fun l => N l + 1) (t + 1) l) :=
      neg_one_pow_of_step (sum_of_eq_off hoth (fun _ x => x)) hstep
    simp only [ryserRepGrayStep, Nat.add_sub_cancel, hu, hv, natIdx_of_lt N hi, hcol]
    refine ⟨hsums, hsgn, hbin, ?_⟩
    rw [hsums, hbin, hsgn, sum_range_succ _ (t + 1)]
  rw [ryserRepGray, hP.2.2.2, Nat.sub_add_cancel hK]
  have := sum_range_naryGray (fun l => N l + 1) term
  simp only at this
  rw [this, ryserRep, mul_sum]
  refine sum_congr rfl fun s hs => ?_
  have hle : ∑ l, s l ≤ ∑ l, N l := by
    apply sum_le_sum
    intro l _
    have := Fintype.mem_piFinset.mp hs l
    simp only [mem_range] at this
    omega
  simp only [term, rs]
  rw [neg_one_pow_sub hle]
  ring

/-! ## n-ary Gray code: BB/FG with multiplicities -/

theorem glynnRepGray_eq_glynnRep {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (k0 : Fin p)
    (M : Fin p → ℕ) (N : Fin q → ℕ) : glynnRepGray A k0 M N = glynnRep A k0 M N := by
  let cs : (Fin p → ℕ) → Fin q → R := fun d l =>
    A k0 l + ∑ k, ((M k : R) - 2 * (d k : R)) * A k l
  let term : (Fin p → ℕ) → R := fun d =>
    (-1) ^ (∑ k, d k) * ((∏ k, (M k).choose (d k) : ℕ) : R) * ∏ l, cs d l ^ N l
  let P : ℕ → RepState (Fin q) R → Prop := fun t st =>
    st.sums = cs (naryGray (fun k => M k + 1) t) ∧
      st.sign = (-1) ^ (∑ k, naryGray (fun k => M k + 1) t k) ∧
      st.binom = ∏ k, (M k).choose (naryGray (fun k => M k + 1) t k) ∧
      st.acc = ∑ u ∈ range (t + 1), term (naryGray (fun k => M k + 1) u)
  have hK := one_le_prod_succ M
  have hinit : (fun l => A k0 l + ∑ k, (M k : R) * A k l) = cs (naryGray (fun k => M k + 1) 0) := by
    funext l; simp [cs, naryGray_zero]
  have hP := runSteps_induction (glynnRepGrayStep A M N)
    ⟨fun l => A k0 l + ∑ k, (M k : R) * A k l, 1, 1,
      ∏ l, (A k0 l + ∑ k, (M k : R) * A k l) ^ N l⟩ P
    (∏ k, (M k + 1) - 1) ?base ?step _ le_rfl
  case base =>
    refine ⟨hinit, by simp [naryGray_zero], by simp [naryGray_zero], ?_⟩
    simp only [zero_add, range_one, sum_singleton, term]
    rw [← hinit]
    simp [naryGray_zero]
  case step =>
    rintro t ⟨sums, sign, binom, acc⟩ ht ⟨hsums, hsign, hbinom, hacc⟩
    simp only at hsums hsign hbinom hacc
    subst hsums hsign hbinom hacc
    have ht' : t + 1 < ∏ k, (M k + 1) := by omega
    obtain ⟨hi, hoth, hstep, hu, hv⟩ := naryGray_step_data (fun k => M k + 1) ht'
    rw [hu, hv] at hstep
    have hul : naryGray (fun k => M k + 1) t ⟨_, hi⟩ ≤ M ⟨_, hi⟩ :=
      Nat.lt_succ_iff.mp (naryGray_lt _ (by omega) _)
    have hvl : naryGray (fun k => M k + 1) (t + 1) ⟨_, hi⟩ ≤ M ⟨_, hi⟩ :=
      Nat.lt_succ_iff.mp (naryGray_lt _ ht' _)
    have hrow : natIdx A (naryGrayIdx (fun k => M k + 1) t) = (fun k l => A k l) ⟨_, hi⟩ := by
      unfold natIdx; rw [dif_pos hi]
    have hsums := cs_step (R := R) hoth hstep (fun l => A k0 l) M (fun k l => A k l)
    have hbin := binom_step M hoth hstep hul hvl
    have hsgn : -(-1 : R) ^ (∑ k, naryGray (fun k => M k + 1) t k) =
        (-1) ^ (∑ k, naryGray (fun k => M k + 1) (t + 1) k) :=
      neg_one_pow_of_step (sum_of_eq_off hoth (fun _ x => x)) hstep
    simp only [glynnRepGrayStep, Nat.add_sub_cancel, hu, hv, natIdx_of_lt M hi, hrow]
    refine ⟨hsums, hsgn, hbin, ?_⟩
    rw [hsums, hbin, hsgn, sum_range_succ _ (t + 1)]
  rw [glynnRepGray, hP.2.2.2, Nat.sub_add_cancel hK]
  exact sum_range_naryGray (fun k => M k + 1) term

end MaxelerPermanent
