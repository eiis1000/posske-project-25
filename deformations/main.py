from .sage_proxy import sa  # initializes properly

from .algebra import GlobalAlgebra
from .charges import ShortRangeChain
from .operators.gln import make_gln, GLNBoostOp


def main():
    alg = GlobalAlgebra(GLNBoostOp.boost, make=make_gln)
    i_ = alg.i_  # keep for interactive

    # test = GlobalPermOp([1, 3, 2])
    # print("Test permutation:", test)

    hamiltonian = alg.make([1]) - alg.make([2, 1])
    chain = ShortRangeChain(hamiltonian, logging=True)
    chain.ensure_filled(6)

    deformation_top = 6
    deformation_target = 2
    deformations = [None, None]
    for k in range(2, deformation_top + 1):
        # Q3_deformations.append(i_ * chain.BQ(3).bracket(chain.Q(k)) )
        deformations.append(
            chain.first_order_boost_deformation_reduced(deformation_target, k)
        )
        print(f"Q_{deformation_target}^[{k}]: {deformations[k]}\n----")

    # TODO I think it would be much nicer to have homog, boost, bilocal all
    # have the permutations as raw data and have the constructor do the
    # operator identifications so that all of that tooling can be pulled out
    # of the bracket. this also substantially simplifies the bracket logic
    # because it means that I just make boosts in the bracket and they should
    # automatically cancel, same for bilocals, etc.

    Q, BQ = chain.Q, chain.BQ
    breakpoint()


if __name__ == "__main__":
    main()
