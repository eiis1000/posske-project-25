from .algebra import GlobalAlgebra
from .long_chains import LongRangeChain
from .operators.gln import GLNBilocalOp, GLNBoostOp, GLNHomogOp, make_gln
from .sage_proxy import sa  # initializes properly
from .short_chains import ShortRangeChain


def main():
    # definitions for repl convenience

    alg = GlobalAlgebra(GLNBoostOp.boost, GLNBilocalOp.bilocalize, make=make_gln)
    # alg = GlobalAlgebra(None, GLNBilocalOp.bilocalize, make=make_gln) # bilocal boost
    i_ = alg.i_  # keep for interactive

    # test_perm = alg.make([1, 3, 4, 2])
    # print("Test permutation:", test_perm)

    hamiltonian = alg.make([1]) - alg.make([2, 1])
    # print("Test commutator:", alg.bracket(test_perm, hamiltonian))

    chain = ShortRangeChain(hamiltonian)  # , logging=True)
    # chain.ensure_filled(6)

    # deformation_top = 6
    # deformation_target = 2
    # deformations = [None, None]
    # for k in range(2, deformation_top + 1):
    #     deformations.append(
    #         chain.first_order_boost_deformation_reduced(deformation_target, k)
    #     )
    #     print(f"Q_{deformation_target}^[{k}]: {deformations[k]}")  # \n----")

    # Q, BQ = chain.Q, chain.BQ
    # lbl = lambda a, b: alg.bilocalize(Q(a), Q(b))
    # BQ2 = BQ(2)
    # BLBQ2 = bilocal_boost(Q(2))
    # print("eq. 3.34: 1/4*(", i_ * 4 * lbl(2, 3).bracket(Q(2)), ")")

    # # if a test passes everything up to and incl order K fill Q, I will write
    # # "P(K;Q)". if it fails on homog, I'll write "H(K;Q)", and if it fails on
    # # algebra consistency I'll write "C(K;Q)". if it disagrees with Table 2,
    # # I'll write "T(K)" (since Q=2 always). if no ensure_filled is required for
    # # failure, I'll just write e.g. H(K), or I'll write H(K;_Q) where _Q is the
    # # charge that fails the test.
    # ## here are the boosts
    # # deform = (2,)    # P(5;4), consistency too slow
    # # deform = (3,)    # H(4;_2)
    # # deform = (4,)    # H(4;_2)
    # ## here are the boosts again, but this time generated bilocally
    # # deform = (2, -1)  # P(5;4)
    # # deform = (3, -1)  # P(5;4), consistency too slow
    # ## here are the bilocals
    # # deform = (1, 3)  # P(4)
    # deform = (2, 3)  # C(1)
    # # deform = (2, 4)  # T(1)
    # lrc = LongRangeChain(chain, deform)
    # lrc.ensure_order(1)
    # # lrc.ensure_order(3)
    # print("Q2: ", lrc.format(lrc.Q(2)))
    # print("Q3: ", lrc.format(lrc.Q(3)))
    # # print("Q4: ", lrc.format(lrc.Q(4)))
    # print("-" * 20)
    # print(
    #     "[Q3, Q2]: ",
    #     lrc.format(lrc.bracket_to_order(lrc.Q(3), lrc.Q(2)), lrc.order()),
    # )
    # # lrc.bracket_to_order(lrc.deform_gen(), lrc.Q(2))
    # # print("All brackets consistent: ", lrc.algebra_consistency())

    # # lrc.ensure_order(3)
    # # lrc.ensure_filled(4)
    # # print("[Q3, Q4]: ", lrc.format(lrc.bracket_to_order(Q(3), Q(4), 2), 2))
    # # print(
    # #     "[B[Q3], Q4]: ",
    # #     str(lrc.bracket_to_order(alg.boost(Q(3)), Q(4), 2), 2)[:100],
    # #     " + ...",
    # # )
    # Q = lrc.Q
    breakpoint()


import pdb
import sys


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type == AssertionError:
        print("AssertionError caught, dropping into debugger:")
        pdb.post_mortem(exc_traceback)
    else:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = custom_excepthook

if __name__ == "__main__":
    main()
