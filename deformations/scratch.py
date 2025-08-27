from .algebra import GlobalAlgebra
from .deformed_chains import DeformedChain
from .operators.gln import GLNBilocalOp, make_gln
from .spin_chains import BaseChain


def algebra_scratch():
    alg = GlobalAlgebra(GLNBilocalOp.bilocalize, make=make_gln)
    test_perm = alg.make([1, 3, 4, 2])
    test_element = alg.make([1]) - alg.make([3, 2, 1]) + 2 * alg.make([2, 1, 4, 3])
    test_boost = alg.boost(test_perm)
    test_bilocal = alg.bilocalize(test_perm, test_element)
    print("Test permutation:", test_perm)
    print("Test commutator 1:", alg.bracket(test_perm, test_element))
    print("Test commutator 2:", alg.bracket(test_boost, test_element))
    print("Test commutator 3:", alg.bracket(test_bilocal, test_element))


def deformation_scratch(bp=False):
    alg = GlobalAlgebra(GLNBilocalOp.bilocalize, make=make_gln)
    hamiltonian = alg.make([1]) - alg.make([2, 1])
    base_chain = BaseChain(hamiltonian)
    deformed = DeformedChain(base_chain, (1, 3))  # bilocal of [1] and Q3
    deformed.ensure_order(3)
    print("Q2: ", deformed.format(deformed.Q(2)))
    print("Q3: ", deformed.format(deformed.Q(3)))
    print("-" * 20)
    print("[Q3, Q2]: ", deformed.format(deformed.bracket(deformed.Q(3), deformed.Q(2))))
    print("All brackets consistent: ", deformed.algebra_consistency())
    if bp:
        breakpoint()


def main():
    deformation_scratch(bp=True)


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
