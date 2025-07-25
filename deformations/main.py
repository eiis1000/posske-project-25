from .algebra import GlobalGLNAlgebra
from .charges import ShortRangeChain
from .operators import (
    GlobalPermOp,
    GlobalBoostOp,
)


def main():
    alg = GlobalGLNAlgebra()
    i_ = alg.i_

    # test = GlobalPermOp([1, 3, 2])
    # print("Test permutation:", test)

    def GPO(perm):
        return alg(GlobalPermOp(perm))

    hamiltonian = GPO([1]) - GPO([2, 1])
    chain = ShortRangeChain(hamiltonian, GlobalBoostOp.boost, logging=True)
    chain.ensure_filled(6)

    deformation_top = 5
    deformation_target = 2
    Q3_deformations = [None, None]
    for k in range(2, deformation_top + 1):
        # Q3_deformations.append(i_ * chain.BQ(3).bracket(chain.Q(k)) )
        Q3_deformations.append(
            chain.first_order_boost_deformation_reduced(deformation_target, k)
        )
        print(f"Q_{deformation_target}^[{k}]: {Q3_deformations[k]}\n----")

    breakpoint()


if __name__ == "__main__":
    main()
