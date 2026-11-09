module

public import MaxelerPermanent

/-!
# Proved statements

Each declaration below has exactly the statement of its counterpart in
`Challenge.lean`; `lake comparator` checks this and that only the standard axioms
are used. The proofs live in the `MaxelerPermanent` library.
-/

public section

namespace MaxelerPermanent

variable {R : Type*} [CommRing R]

theorem permNaive_correct {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    permNaive A = A.permanent :=
  permNaive_eq_permanent A

theorem detNaive_correct {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    detNaive A = A.det :=
  detNaive_eq_det A

theorem ryser_correct {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    ryser A = A.permanent :=
  ryser_eq_permanent A

theorem glynn_correct {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynn A = 2 ^ n * A.permanent :=
  glynn_eq A

theorem ryserGray_correct {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    ryserGray A = A.permanent :=
  ryserGray_eq_permanent A

theorem glynnGray_correct {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynnGray A = 2 ^ n * A.permanent :=
  glynnGray_eq A

theorem ryserRep_correct {n p q : ℕ} (A : Matrix (Fin p) (Fin q) R)
    (r : Fin n → Fin p) (c : Fin n → Fin q) :
    (A.submatrix r c).permanent = ryserRep A (mult r) (mult c) :=
  ryserRep_eq_permanent A r c

theorem glynnRep_correct {n p q : ℕ} (A : Matrix (Fin p) (Fin q) R)
    (r : Fin (n + 1) → Fin p) (c : Fin (n + 1) → Fin q) :
    2 ^ n * (A.submatrix r c).permanent =
      glynnRep A (r 0) (mult (r ∘ Fin.succ)) (mult c) :=
  glynnRep_eq A r c

theorem ryserRepGray_correct {n p q : ℕ} (A : Matrix (Fin p) (Fin q) R)
    (r : Fin n → Fin p) (c : Fin n → Fin q) :
    (A.submatrix r c).permanent = ryserRepGray A (mult r) (mult c) := by
  rw [ryserRepGray_eq_ryserRep]
  exact ryserRep_eq_permanent A r c

theorem glynnRepGray_correct {n p q : ℕ} (A : Matrix (Fin p) (Fin q) R)
    (r : Fin (n + 1) → Fin p) (c : Fin (n + 1) → Fin q) :
    2 ^ n * (A.submatrix r c).permanent =
      glynnRepGray A (r 0) (mult (r ∘ Fin.succ)) (mult c) := by
  rw [glynnRepGray_eq_glynnRep]
  exact glynnRep_eq A r c

theorem glynnWiki_correct {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynnWiki A = 2 ^ n * A.permanent :=
  glynnWiki_eq A

theorem permNaive_complexity {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    permNaiveC A = ⟨permNaive A, n.factorial * (n + 1)⟩ :=
  permNaiveC_eq A

theorem detNaive_complexity {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    detNaiveC A = ⟨detNaive A, n.factorial * (n + 2)⟩ :=
  detNaiveC_eq A

theorem ryser_complexity {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    ryserC A = ⟨ryser A, n ^ 2 * 2 ^ (n - 1) + (n + 2) * 2 ^ n⟩ :=
  ryserC_eq A

theorem glynn_complexity {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynnC A = ⟨glynn A, 2 ^ n * ((n + 1) * (2 * n + 2) + 2)⟩ :=
  glynnC_eq A

theorem ryserGray_complexity {n : ℕ} (A : Matrix (Fin n) (Fin n) R) :
    ryserGrayC A = ⟨ryserGray A, (2 ^ n - 1) * (2 * n + 3) + n + 1⟩ :=
  ryserGrayC_eq A

theorem glynnGray_complexity {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynnGrayC A = ⟨glynnGray A, (2 ^ n - 1) * (3 * n + 6) + (n + 1) ^ 2 + (n + 1)⟩ :=
  glynnGrayC_eq A

theorem glynnWiki_complexity {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    glynnWikiC A = ⟨glynnWiki A, (2 ^ n - 1) * (3 * n + 4) + (n + 1) ^ 2 + (n + 1)⟩ :=
  glynnWikiC_eq A

theorem ryserRep_complexity {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (M : Fin p → ℕ)
    (N : Fin q → ℕ) :
    ryserRepC A M N = ⟨ryserRep A M N,
      (∏ l, (N l + 1)) * (2 * p * q + ∑ k, M k + p + 3)⟩ :=
  ryserRepC_eq A M N

theorem glynnRep_complexity {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (k0 : Fin p)
    (M : Fin p → ℕ) (N : Fin q → ℕ) :
    glynnRepC A k0 M N = ⟨glynnRep A k0 M N,
      (∏ k, (M k + 1)) * (2 * p * q + 2 * q + ∑ l, N l + 3)⟩ :=
  glynnRepC_eq A k0 M N

theorem ryserRepGray_complexity {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (M : Fin p → ℕ)
    (N : Fin q → ℕ) :
    ryserRepGrayC A M N = ⟨ryserRepGray A M N,
      (∏ l, (N l + 1) - 1) * (2 * p + ∑ k, M k + 4) + ∑ k, M k + p + 1⟩ :=
  ryserRepGrayC_eq A M N

theorem glynnRepGray_complexity {p q : ℕ} (A : Matrix (Fin p) (Fin q) R) (k0 : Fin p)
    (M : Fin p → ℕ) (N : Fin q → ℕ) :
    glynnRepGrayC A k0 M N = ⟨glynnRepGray A k0 M N,
      (∏ k, (M k + 1) - 1) * (3 * q + ∑ l, N l + 4) + q * (2 * p + 1) + ∑ l, N l + q⟩ :=
  glynnRepGrayC_eq A k0 M N

theorem ryserShift_correct {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    ryserShift A = 2 ^ n * A.permanent :=
  ryserShift_eq A

theorem ryserShiftGray_correct {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    ryserShiftGray A = 2 ^ n * A.permanent :=
  ryserShiftGray_eq A

theorem ryserShift_complexity {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    ryserShiftC A = ⟨ryserShift A, 2 ^ n * ((n + 1) * (n + 6) + 2) + (n + 1) * (n * 2 ^ (n - 1))⟩ :=
  ryserShiftC_eq A

theorem ryserShiftGray_complexity {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) R) :
    ryserShiftGrayC A = ⟨ryserShiftGray A, (2 ^ n - 1) * (3 * n + 6) + (n + 1) * (n + 5)⟩ :=
  ryserShiftGrayC_eq A

section Field

variable {K : Type*} [Field K]

theorem ryserNW_correct {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K) (h2 : (2 : K) ≠ 0) :
    ryserNW A = A.permanent :=
  ryserNW_eq A h2

theorem glynnNorm_correct {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K) (r : Fin (n + 1) → K)
    (hr : ∀ i, r i ≠ 0) :
    glynnNorm A r = 2 ^ n * A.permanent :=
  glynnNorm_eq A r hr

theorem ryserNW_complexity {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K) :
    ryserNWC A =
      ⟨ryserNW A, 2 ^ n * ((n + 1) * (n + 5) + 2) + (n + 1) * (n * 2 ^ (n - 1)) + 2⟩ :=
  ryserNWC_eq A

theorem glynnNorm_complexity {n : ℕ} (A : Matrix (Fin (n + 1)) (Fin (n + 1)) K) (r : Fin (n + 1) → K) :
    glynnNormC A r = ⟨glynnNorm A r,
      (2 ^ n - 1) * (3 * n + 4) + 2 * (n + 1) ^ 2 + 2 * (n + 1) + 1⟩ :=
  glynnNormC_eq A r

end Field

end MaxelerPermanent
