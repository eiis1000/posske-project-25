from .sage_proxy import sa  # initializes properly

from .algebra import GlobalAlgebra
from .charges import ShortRangeChain
from .operators.gln import make_gln, GLNBoostOp, GLNBilocalOp


def main():
    alg = GlobalAlgebra(GLNBoostOp.boost, GLNBilocalOp.bilocalize, make=make_gln)
    i_ = alg.i_  # keep for interactive

    # test_perm = alg.make([1, 3, 4, 2])
    # print("Test permutation:", test_perm)

    hamiltonian = alg.make([1]) - alg.make([2, 1])
    # print("Test commutator:", alg.bracket(test_perm, hamiltonian))

    chain = ShortRangeChain(hamiltonian, logging=True)
    chain.ensure_filled(6)

    deformation_top = 4
    deformation_target = 2
    deformations = [None, None]
    for k in range(2, deformation_top + 1):
        deformations.append(
            chain.first_order_boost_deformation_reduced(deformation_target, k)
        )
        print(f"Q_{deformation_target}^[{k}]: {deformations[k]}")  # \n----")

    def bilocal_boost(q):
        ones = chain.Q(1)
        return (alg.bilocalize(ones, q) - alg.bilocalize(q, ones)) / 2

    # definitions for repl convenience
    Q, BQ = chain.Q, chain.BQ
    lbl = lambda a, b: alg.bilocalize(Q(a), Q(b))
    BQ2 = BQ(2)
    BLBQ2 = bilocal_boost(Q(2))
    print("eq. 3.34: 1/4*(", i_ * 4 * lbl(2, 3).bracket(Q(2)), ")")
    breakpoint()


if __name__ == "__main__":
    main()
