from .algebra import GlobalAlgebra
from .charges import ShortRangeChain
from .operators.gln import make_gln, GLNBoostOp
import sage.all as sa  # keep for interactive


def main():
    alg = GlobalAlgebra(GLNBoostOp.boost, make=make_gln)
    i_ = alg.i_  # keep for interactive

    # test = GlobalPermOp([1, 3, 2])
    # print("Test permutation:", test)

    hamiltonian = alg.make([1]) - alg.make([2, 1])
    chain = ShortRangeChain(hamiltonian, logging=True)
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
