module

public import MaxelerPermanent.Defs
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Data.Fintype.BigOperators

/-!
# Gray codes

Properties of the reflected mixed-radix Gray code `naryGray` and of the binary
reflected Gray code `grayCode k = k ⊕ (k >> 1)`:

* `naryGray_lt`: for `t < ∏ rad`, every digit is in range;
* `naryGray_injOn`: the codes are distinct, hence `sum_range_naryGray` re-indexes a
  sum over the box `∏_k [0, rad k)` as a sum over `t < ∏ rad`;
* `naryGray_succ`: consecutive codes differ in the single digit `naryGrayIdx rad t`,
  by exactly one;
* `grayCode_testBit`, `trailingZeros_succ`: the binary code is the radix-2 case,
  and `trailingZeros (t + 1)` is the bit that changes.
-/

public section

open Finset

namespace MaxelerPermanent

/-- The box `∏_k [0, rad k)` of digit vectors. -/
abbrev box {m : ℕ} (rad : Fin m → ℕ) : Finset (Fin m → ℕ) :=
  Fintype.piFinset fun k => range (rad k)

theorem card_box {m : ℕ} (rad : Fin m → ℕ) : #(box rad) = ∏ k, rad k := by
  simp [box]

theorem lt_prod_aux {m : ℕ} {rad : Fin (m + 1) → ℕ} {t : ℕ} (ht : t < ∏ k, rad k) :
    0 < rad 0 ∧ t / rad 0 < ∏ k, Fin.tail rad k := by
  rw [Fin.prod_univ_succ] at ht
  have h0 : 0 < rad 0 := by
    rcases Nat.eq_zero_or_pos (rad 0) with h | h
    · rw [h, zero_mul] at ht; omega
    · exact h
  exact ⟨h0, (Nat.div_lt_iff_lt_mul h0).mpr (by rw [mul_comm]; exact ht)⟩

theorem naryGray_zero_digit {m : ℕ} (rad : Fin (m + 1) → ℕ) (t : ℕ) :
    naryGray rad t 0 =
      if t / rad 0 % 2 = 0 then t % rad 0 else rad 0 - 1 - t % rad 0 := by
  simp [naryGray]

theorem naryGray_succ_digit {m : ℕ} (rad : Fin (m + 1) → ℕ) (t : ℕ) (k : Fin m) :
    naryGray rad t k.succ = naryGray (Fin.tail rad) (t / rad 0) k := by
  simp [naryGray]

theorem naryGray_lt {m : ℕ} (rad : Fin m → ℕ) {t : ℕ} (ht : t < ∏ k, rad k) (k : Fin m) :
    naryGray rad t k < rad k := by
  induction m generalizing t with
  | zero => exact k.elim0
  | succ m ih =>
    obtain ⟨h0, hq⟩ := lt_prod_aux ht
    refine Fin.cases ?_ (fun k => ?_) k
    · rw [naryGray_zero_digit]
      have := Nat.mod_lt t h0
      split_ifs <;> omega
    · rw [naryGray_succ_digit]
      exact ih (Fin.tail rad) hq k

theorem naryGray_mem_box {m : ℕ} (rad : Fin m → ℕ) {t : ℕ} (ht : t < ∏ k, rad k) :
    naryGray rad t ∈ box rad := by
  simp only [box, Fintype.mem_piFinset, mem_range]
  exact naryGray_lt rad ht

theorem naryGray_injOn {m : ℕ} (rad : Fin m → ℕ) :
    Set.InjOn (naryGray rad) (range (∏ k, rad k) : Set ℕ) := by
  induction m with
  | zero =>
    intro a ha b hb _
    simp only [coe_range, Set.mem_Iio, univ_eq_empty, prod_empty] at ha hb
    omega
  | succ m ih =>
    intro a ha b hb hab
    simp only [coe_range, Set.mem_Iio] at ha hb
    obtain ⟨h0, hqa⟩ := lt_prod_aux ha
    obtain ⟨-, hqb⟩ := lt_prod_aux hb
    have htail : a / rad 0 = b / rad 0 := by
      apply ih (Fin.tail rad) (by simpa using hqa) (by simpa using hqb)
      funext k
      rw [← naryGray_succ_digit, ← naryGray_succ_digit, hab]
    have hd := congrFun hab 0
    rw [naryGray_zero_digit, naryGray_zero_digit, htail] at hd
    have hma := Nat.mod_lt a h0
    have hmb := Nat.mod_lt b h0
    have hmod : a % rad 0 = b % rad 0 := by split_ifs at hd <;> omega
    rw [← Nat.div_add_mod a (rad 0), ← Nat.div_add_mod b (rad 0), htail, hmod]

theorem image_naryGray {m : ℕ} (rad : Fin m → ℕ) :
    (range (∏ k, rad k)).image (naryGray rad) = box rad := by
  apply eq_of_subset_of_card_le
  · intro d hd
    obtain ⟨t, ht, rfl⟩ := mem_image.mp hd
    exact naryGray_mem_box rad (mem_range.mp ht)
  · rw [card_image_of_injOn (naryGray_injOn rad), card_range, card_box]

/-- A sum over the box is a sum over the Gray-code enumeration. -/
theorem sum_range_naryGray {m : ℕ} {M : Type*} [AddCommMonoid M] (rad : Fin m → ℕ)
    (F : (Fin m → ℕ) → M) :
    ∑ t ∈ range (∏ k, rad k), F (naryGray rad t) = ∑ d ∈ box rad, F d := by
  rw [← image_naryGray, sum_image (naryGray_injOn rad)]

theorem succ_div_mod_of_lt {r t : ℕ} (h0 : 0 < r) (h : t % r + 1 < r) :
    (t + 1) / r = t / r ∧ (t + 1) % r = t % r + 1 := by
  have e : t + 1 = (t % r + 1) + r * (t / r) := by have := Nat.mod_add_div t r; omega
  rw [e, Nat.add_mul_div_left _ _ h0, Nat.div_eq_of_lt h, zero_add,
    Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt h]
  exact ⟨rfl, rfl⟩

theorem succ_div_mod_of_eq {r t : ℕ} (h0 : 0 < r) (h : t % r + 1 = r) :
    (t + 1) / r = t / r + 1 ∧ (t + 1) % r = 0 := by
  have e : t + 1 = r * (t / r + 1) := by have := Nat.mod_add_div t r; rw [mul_add]; omega
  rw [e, Nat.mul_div_cancel_left _ h0, Nat.mul_mod_right]
  exact ⟨rfl, rfl⟩

/-- Consecutive n-ary Gray codes differ in exactly the digit `naryGrayIdx rad t`,
and there by exactly one. -/
theorem naryGray_succ {m : ℕ} (rad : Fin m → ℕ) {t : ℕ} (ht : t + 1 < ∏ k, rad k) :
    ∃ h : naryGrayIdx rad t < m,
      (∀ k : Fin m, k ≠ ⟨_, h⟩ → naryGray rad (t + 1) k = naryGray rad t k) ∧
      (naryGray rad (t + 1) ⟨_, h⟩ + 1 = naryGray rad t ⟨_, h⟩ ∨
        naryGray rad t ⟨_, h⟩ + 1 = naryGray rad (t + 1) ⟨_, h⟩) := by
  induction m generalizing t with
  | zero => simp at ht
  | succ m ih =>
    obtain ⟨h0, hq⟩ := lt_prod_aux ht
    have hmod := Nat.mod_lt t h0
    by_cases hlast : t % rad 0 + 1 < rad 0
    · -- the fastest digit moves; the others stay
      have hidx : naryGrayIdx rad t = 0 := by simp [naryGrayIdx, hlast]
      obtain ⟨hdiv, hmod'⟩ := succ_div_mod_of_lt h0 hlast
      refine ⟨by omega, ?_, ?_⟩
      · intro k hk
        refine Fin.cases (fun h => absurd (Fin.ext (by simp [hidx])) h) (fun k _ => ?_) k hk
        rw [naryGray_succ_digit, naryGray_succ_digit, hdiv]
      · simp only [hidx]
        rw [show (⟨0, _⟩ : Fin (m + 1)) = 0 from rfl, naryGray_zero_digit,
          naryGray_zero_digit, hdiv, hmod']
        split_ifs <;> omega
    · -- the fastest digit is at its turning point; a slower digit moves
      have hlast' : t % rad 0 + 1 = rad 0 := by omega
      obtain ⟨hdiv, hmod'⟩ := succ_div_mod_of_eq h0 hlast'
      have hq1 : t / rad 0 + 1 < ∏ k, Fin.tail rad k := by
        have := lt_prod_aux (t := t + 1) ht
        rw [hdiv] at this
        exact this.2
      obtain ⟨hi, hother, hstep⟩ := ih (Fin.tail rad) hq1
      have hidx : naryGrayIdx rad t = naryGrayIdx (Fin.tail rad) (t / rad 0) + 1 := by
        simp [naryGrayIdx, hlast]
      refine ⟨by omega, ?_, ?_⟩
      · intro k hk
        refine Fin.cases (fun _ => ?_) (fun k hk => ?_) k hk
        · rw [naryGray_zero_digit, naryGray_zero_digit, hdiv, hmod']
          have := Nat.mod_two_eq_zero_or_one (t / rad 0)
          split_ifs <;> omega
        · rw [naryGray_succ_digit, naryGray_succ_digit, hdiv]
          apply hother
          intro hk'
          apply hk
          ext
          simp [hidx]
          exact congrArg Fin.val hk'
      · have e : (⟨naryGrayIdx rad t, by omega⟩ : Fin (m + 1)) = Fin.succ ⟨_, hi⟩ := by
          ext; simp [hidx]
        rw [e, naryGray_succ_digit, naryGray_succ_digit, hdiv]
        exact hstep

/-! ## Binary Gray code -/

theorem grayCode_testBit_zero (t : ℕ) :
    (grayCode t).testBit 0 = decide ((t % 2 = 1) ≠ (t / 2 % 2 = 1)) := by
  rw [grayCode, Nat.testBit_xor, Nat.testBit_shiftRight, show 1 + 0 = 0 + 1 from rfl,
    Nat.testBit_succ, Nat.testBit_zero, Nat.testBit_zero]
  rcases Nat.mod_two_eq_zero_or_one t with h | h <;>
    rcases Nat.mod_two_eq_zero_or_one (t / 2) with h' | h' <;> simp [h, h']

theorem grayCode_testBit_succ (t i : ℕ) :
    (grayCode t).testBit (i + 1) = (grayCode (t / 2)).testBit i := by
  simp only [grayCode, Nat.testBit_xor, Nat.testBit_shiftRight, Nat.testBit_div_two]
  congr 1 <;> congr 1 <;> omega

/-- The binary reflected Gray code is the radix-2 case of `naryGray`. -/
theorem grayCode_testBit {m : ℕ} (t : ℕ) (i : Fin m) :
    (grayCode t).testBit i = decide (naryGray (fun _ => 2) t i = 1) := by
  induction m generalizing t with
  | zero => exact i.elim0
  | succ m ih =>
    refine Fin.cases ?_ (fun i => ?_) i
    · rw [naryGray_zero_digit, Fin.val_zero, grayCode_testBit_zero]
      rcases Nat.mod_two_eq_zero_or_one t with h | h <;>
        rcases Nat.mod_two_eq_zero_or_one (t / 2) with h' | h' <;> simp [h, h']
    · rw [naryGray_succ_digit, Fin.val_succ, grayCode_testBit_succ]
      exact ih (t / 2) i

theorem trailingZeros_succ {m : ℕ} {t : ℕ} (ht : t + 1 < 2 ^ m) :
    trailingZeros (t + 1) = naryGrayIdx (fun _ : Fin m => 2) t := by
  induction m generalizing t with
  | zero => simp at ht
  | succ m ih =>
    rw [trailingZeros]
    simp only [Nat.add_one_ne_zero, ↓reduceDIte]
    rcases Nat.mod_two_eq_zero_or_one t with h | h
    · have : (t + 1) % 2 = 1 := by omega
      simp [naryGrayIdx, this, h]
    · have h1 : (t + 1) % 2 ≠ 1 := by omega
      have h2 : ¬ (t % 2 + 1 < 2) := by omega
      simp only [h1, ↓reduceIte, naryGrayIdx, h2]
      have : (t + 1) / 2 = t / 2 + 1 := by omega
      rw [this, ih (by rw [pow_succ] at ht; omega)]
      rfl

theorem naryGray_two_le_one {m : ℕ} {t : ℕ} (ht : t < 2 ^ m) (k : Fin m) :
    naryGray (fun _ => 2) t k ≤ 1 := by
  have := naryGray_lt (fun _ : Fin m => 2) (t := t) (by simpa using ht) k
  omega

/-- The subset of `Fin m` encoded by the bits of `grayCode t`. -/
def graySet (m t : ℕ) : Finset (Fin m) := {i | (grayCode t).testBit i}

theorem mem_graySet {m t : ℕ} (i : Fin m) :
    i ∈ graySet m t ↔ naryGray (fun _ => 2) t i = 1 := by
  simp [graySet, grayCode_testBit]

theorem graySet_zero (m : ℕ) : graySet m 0 = ∅ := by
  ext i; simp [graySet, grayCode]

theorem graySet_injOn (m : ℕ) : Set.InjOn (graySet m) (range (2 ^ m) : Set ℕ) := by
  intro a ha b hb hab
  have hrange : (range (2 ^ m) : Set ℕ) = (range (∏ _k : Fin m, 2) : Set ℕ) := by simp
  rw [hrange] at ha hb
  apply naryGray_injOn (fun _ : Fin m => 2) ha hb
  funext k
  have hk := congrArg (k ∈ ·) hab
  simp only [mem_graySet, eq_iff_iff] at hk
  have h1 := naryGray_two_le_one (by simpa using ha) k
  have h2 := naryGray_two_le_one (by simpa using hb) k
  omega

/-- A sum over all subsets of `Fin m` is a sum over the binary Gray-code enumeration. -/
theorem sum_range_graySet {m : ℕ} {M : Type*} [AddCommMonoid M]
    (F : Finset (Fin m) → M) :
    ∑ t ∈ range (2 ^ m), F (graySet m t) = ∑ S : Finset (Fin m), F S := by
  rw [← sum_image (graySet_injOn m)]
  congr 1
  apply eq_of_subset_of_card_le (subset_univ _)
  rw [card_image_of_injOn (graySet_injOn m), card_range, card_univ, Fintype.card_finset,
    Fintype.card_fin]

/-- From step `t` to `t + 1`, `graySet` toggles exactly the element `trailingZeros (t+1)`,
which is inserted iff its bit is set in `grayCode (t + 1)`. -/
theorem graySet_succ {m t : ℕ} (ht : t + 1 < 2 ^ m) :
    ∃ h : trailingZeros (t + 1) < m,
      if (grayCode (t + 1)).testBit (trailingZeros (t + 1)) then
        (⟨_, h⟩ : Fin m) ∉ graySet m t ∧ graySet m (t + 1) = insert ⟨_, h⟩ (graySet m t)
      else
        (⟨_, h⟩ : Fin m) ∈ graySet m t ∧ graySet m (t + 1) = (graySet m t).erase ⟨_, h⟩ := by
  have hp : t + 1 < ∏ _k : Fin m, 2 := by simpa using ht
  obtain ⟨hi, hother, hstep⟩ := naryGray_succ (fun _ : Fin m => 2) hp
  have htz := trailingZeros_succ ht
  have hi' : trailingZeros (t + 1) < m := htz ▸ hi
  refine ⟨hi', ?_⟩
  have e : (⟨trailingZeros (t + 1), hi'⟩ : Fin m) = ⟨_, hi⟩ := by ext; exact htz
  have b1 := naryGray_two_le_one (t := t) (by omega) ⟨_, hi⟩
  have b2 := naryGray_two_le_one (t := t + 1) ht ⟨_, hi⟩
  have hbit : (grayCode (t + 1)).testBit (trailingZeros (t + 1)) =
      decide (naryGray (fun _ => 2) (t + 1) ⟨_, hi⟩ = 1) := by
    have := grayCode_testBit (m := m) (t + 1) ⟨_, hi'⟩
    rw [this, e]
  rw [hbit, e]
  have hoth : ∀ k : Fin m, k ≠ ⟨_, hi⟩ → (k ∈ graySet m (t + 1) ↔ k ∈ graySet m t) := by
    intro k hk
    rw [mem_graySet, mem_graySet, hother k hk]
  by_cases hv : naryGray (fun _ => 2) (t + 1) ⟨_, hi⟩ = 1
  · have hu : naryGray (fun _ => 2) t ⟨_, hi⟩ = 0 := by omega
    simp only [hv, decide_true, ↓reduceIte]
    refine ⟨by rw [mem_graySet, hu]; omega, ?_⟩
    ext k
    by_cases hk : k = ⟨_, hi⟩
    · subst hk; simp [mem_graySet, hv]
    · rw [mem_insert, hoth k hk]; simp [hk]
  · have hu : naryGray (fun _ => 2) t ⟨_, hi⟩ = 1 := by omega
    simp only [hv, decide_false, Bool.false_eq_true, ↓reduceIte]
    refine ⟨by rw [mem_graySet, hu], ?_⟩
    ext k
    by_cases hk : k = ⟨_, hi⟩
    · subst hk; simp [mem_graySet, hv]
    · rw [mem_erase, hoth k hk]; simp [hk]

end MaxelerPermanent
