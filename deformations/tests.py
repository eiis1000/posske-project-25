import time
import traceback

import numpy as np

from .algebra import GlobalAlgebra
from .deformed_chains import DeformedChain
from .operators.gln import (
    GLNBilocalOp,
    GLNBoostOp,
    GLNHomogOp,
    make_gln,
    reduce_permutation,
)
from .spin_chains import BaseChain
from .tools import is_zero


def run_deformation_test(deform, max_order, max_q, skip_last_consistency=False):
    match deform:
        case (x,):
            deform_str = f"boost B[Q_{x}]"
        case (x, -1):
            deform_str = f"bilocal-boost B[Q_{x}]"
        case (x, y):
            deform_str = f"bilocal [Q_{x} | Q_{y}]"
        case _:
            deform_str = str(deform)

    test_name = f"deforming with {deform_str} to order {max_order} up to Q_{max_q}"

    start_time = time.perf_counter()
    print(f"Running test {test_name}...")
    try:
        alg = GlobalAlgebra(
            GLNBilocalOp.bilocalize, boost=GLNBoostOp.boost, make=make_gln
        )
        hamiltonian = alg.make([1]) - alg.make([2, 1])
        base_chain = BaseChain(hamiltonian)
        deformed_chain = DeformedChain(base_chain, deform)
        deformed_chain.ensure_filled(max_q)
        # note we could just set ord=max_order, but this way we can check
        # which specific order fails
        consistent = True
        for ord in range(1, max_order + 1):
            print(f"Computing order {ord}...", end="\r")
            deformed_chain.ensure_order(ord)
            if skip_last_consistency and ord == max_order:
                # this option exists for speed reasons. the check is slow
                continue
            print(f"Order {ord} consistency...", end="\r")
            consistent = deformed_chain.algebra_consistency()
            if not consistent:
                print(f"Test {test_name} FAILED on consistency at order {ord}.")
                break
        if consistent:
            print(f"Test {test_name} PASSED.")
    except Exception as e:
        print(f"Test {test_name} FAILED with an exception: {e}")
        traceback.print_exc()
    print(f"Elapsed: {time.perf_counter() - start_time:.2f}s\n")
    return deformed_chain


def deformation_tests():
    print("Starting deformation tests...")

    tests = [
        ((2,), 5, 4),
        ((3,), 4, 2),
        ((4,), 3, 2, True),
        ((2, -1), 5, 4),
        ((3, -1), 3, 4),
        ((1, 3), 3, 4),
        ((1, 3), 4, 3),
        ((2, 3), 3, 2, True),
        ((2, 4), 2, 2),
    ]

    speedy = True
    speedy = False
    for test in tests:
        if speedy:
            test = (test[0], max(1, test[1] - 1), 3)  # , True)
        run_deformation_test(*test)

    print("All deformation tests completed.")
    return True


def jacobi(a, b, c):
    return a.bracket(b.bracket(c)) + b.bracket(c.bracket(a)) + c.bracket(a.bracket(b))


def jacobi_tests():
    print("Running Jacobi tests...")
    alg = GlobalAlgebra(GLNBilocalOp.bilocalize, boost=GLNBoostOp.boost, make=make_gln)
    make = alg.make
    i = "precursor"
    assert alg.zero() == jacobi(
        make([2, 1]),
        make([2, 3, 1]),
        make(([2, 1], [2, 1])),
    ), "Precursor bilocal Jacobi failed"
    assert alg.zero() == jacobi(
        make([2, 1, 4, 3]),
        make([3, 1, 2]),
        make(([2, 1],)),
    ), "Precursor boost Jacobi failed"

    maxlen = 5
    for i in range(10):
        np.random.seed(i)
        lens = np.random.randint(1, maxlen + 1, 5)
        perms = [
            (reduce_permutation(np.random.permutation(k))[0] + 1).tolist() for k in lens
        ]
        h1 = make(perms[0])
        h2 = make(perms[1])
        boosted = make((perms[2],))
        boost_jacobi = jacobi(h1, h2, boosted)
        assert boost_jacobi == alg.zero(), (
            f"Jacobi test {i} failed for {perms[0]}, {perms[1]}, B[{perms[2]}]: {boost_jacobi}"
        )
        bilocaled = make((perms[3], perms[4]))
        bilocal_jacobi = jacobi(h1, h2, bilocaled)
        assert bilocal_jacobi == alg.zero(), (
            f"Jacobi test {i} failed for {perms[0]}, {perms[1]}, [{perms[3]}|{perms[4]}]: {bilocal_jacobi}"
        )

    print("All Jacobi tests passed.")
    return


def trivial_tests():
    print("Running trivial tests...")
    alg = GlobalAlgebra(GLNBilocalOp.bilocalize, boost=GLNBoostOp.boost, make=make_gln)
    h1 = alg.make([1])
    h21 = alg.make([2, 1])
    h312 = alg.make([3, 1, 2])
    b11 = alg.bilocalize(h1, h1)
    b211 = alg.bilocalize(h21, h1)
    b121 = alg.bilocalize(h1, h21)
    b2121 = alg.bilocalize(h21, h21)
    b3121 = alg.bilocalize(h312, h1)
    b31221 = alg.bilocalize(h312, h21)
    b312312 = alg.bilocalize(h312, h312)
    targets = [h21, h312, b11, b211, b121, b2121, b3121, b31221, b312312]
    for t in targets:
        target_identity_bracket = alg.bracket(t, h1)
        assert alg.zero() == target_identity_bracket, (
            f"Identity bracket test failed for {t}: {target_identity_bracket}"
        )
    print("All trivial tests passed.")


def bilocal_boost_conversion(q, alg):
    accum = alg.zero()
    for k, v in q:
        if type(k) is GLNBoostOp:
            kd = k.data
            kh = v * alg(GLNHomogOp(kd, alg=alg))
            accum += alg.bilocal_boost(kh)
        else:
            accum += alg(k) * v
    return accum


def check_conversion_commutativity(left, right, alg, bracket):
    for k1, v1 in left:
        for k2, v2 in right:
            conv_first = bracket(
                bilocal_boost_conversion([(k1, 1)], alg),
                bilocal_boost_conversion([(k2, 1)], alg),
            )
            conv_second = bilocal_boost_conversion(bracket(alg(k1), alg(k2)), alg)
            diff = conv_first - conv_second
            if not is_zero(diff):
                print(f"Conversion non-commutativity found for {k1} and {k2}: {diff}")
                # return False
    return True


def test_bilocal_boost_consistency():
    # XXX deprecated until GLNBoostOp is fixed, and/or should be used to fix it
    print("Running bilocal boost consistency test...")
    alg = GlobalAlgebra(GLNBilocalOp.bilocalize, boost=GLNBoostOp.boost, make=make_gln)
    hamiltonian = alg.make([1]) - alg.make([2, 1])
    base_chain = BaseChain(hamiltonian)
    deformed_chain = DeformedChain(base_chain, (3,))
    for k in range(4):
        print(f"At order {k}...", end="\r")
        deformed_chain.ensure_order(k)
        Q2 = deformed_chain.Q(2)
        Q3 = deformed_chain.Q(3)
        dg = deformed_chain.deform_gen()
        assert check_conversion_commutativity(dg, Q2, alg, deformed_chain.bracket)
        assert check_conversion_commutativity(dg, Q3, alg, deformed_chain.bracket)
    print("Bilocal boost consistency test passed.")


def run_tests():
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1.value, all_threads=True)
    tests = [
        trivial_tests,
        jacobi_tests,
        deformation_tests,
        # test_bilocal_boost_consistency,
    ]
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"Test {test.__name__} failed by raising an exception: {e}")
            traceback.print_exc()
        print("-" * 40)


if __name__ == "__main__":
    run_tests()
