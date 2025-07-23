from .algebra import GlobalGLNAlgebra
from .operators import (
    GlobalPermOp,
    GlobalBoostOp,
)


def main():
    alg = GlobalGLNAlgebra()
    i_ = alg.i_

    test = GlobalPermOp([1, 3, 2])
    print("Test permutation:", test)

    def GPO(perm):
        return alg(GlobalPermOp(perm))

    boost = GlobalBoostOp.boost

    hamiltonian = GPO([1]) - GPO([2, 1])
    BQ2 = boost(hamiltonian)

    charge_top = 5
    charge_tower = [None, None, hamiltonian]
    for k in range(2, charge_top):
        charge_tower.append(BQ2.bracket(charge_tower[-1]) * -i_ / k)
        print(f"Q_{k + 1}: 1/{k}*({k * charge_tower[k + 1]})")

    Q = charge_tower

    BQ3 = boost(Q[3])
    deformation_top = 3
    Q3_deformations = [None, None]
    for k in range(2, deformation_top + 1):
        Q3_deformations.append(i_ * BQ3.bracket(Q[k]))
        print(f"i[B[Q3],Q{k}]: {Q3_deformations[k]}")

    breakpoint()


if __name__ == "__main__":
    main()
