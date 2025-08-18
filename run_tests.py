from deformations.algebra import GlobalAlgebra
from deformations.long_chains import LongRangeChain
from deformations.operators.gln import make_gln, GLNBoostOp, GLNBilocalOp
from deformations.short_chains import ShortRangeChain
import time


def run_test(deform, max_order, max_q, skip_last_consistency=False):
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
        alg = GlobalAlgebra(GLNBoostOp.boost, GLNBilocalOp.bilocalize, make=make_gln)
        hamiltonian = alg.make([1]) - alg.make([2, 1])
        chain = ShortRangeChain(hamiltonian)
        lrc = LongRangeChain(chain, deform)
        lrc.ensure_filled(max_q)
        # note we could just set ord=max_order, but this way we can check
        # which specific order fails
        consistent = True
        for ord in range(1, max_order + 1):
            lrc.ensure_order(ord)
            if skip_last_consistency and ord == max_order:
                # this option exists for speed reasons. the check is slow
                continue
            consistent = lrc.algebra_consistency()
            if not consistent:
                print(f"Test {test_name} FAILED on consistency at order {ord}.")
                break
        if consistent:
            print(f"Test {test_name} PASSED.")
    except Exception as e:
        print(f"Test {test_name} FAILED with an exception: {e}")
    print(f"Elapsed: {time.perf_counter() - start_time:.2f}s\n")


if __name__ == "__main__":
    print("Starting deformation tests...")

    tests = [
        ((2,), 5, 4),
        ((3,), 3, 2),
        ((3,), 4, 2),  # fails
        # ((4,), 4, 2),  # if prev fails this almost certainly does too
        ((2, -1), 5, 4),
        ((3, -1), 5, 4),
        ((1, 3), 3, 4),
        ((1, 3), 4, 2),
        ((2, 3), 1, 2),  # both fail
        ((2, 4), 1, 2),
    ]

    speedy = True
    # speedy = False
    for test in tests:
        if speedy:
            test = (test[0], max(1, test[1] - 1), 3)  # , True)
        run_test(*test)

    print("All deformation tests completed.")
