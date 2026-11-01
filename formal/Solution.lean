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

end MaxelerPermanent
