from ..algebra import GlobalOp
from ..config import perm_print_joiner, symmetric_boost_identification
from ..tools import extend_linear, anticomm

import numpy as np
import sage.all as sa


def is_valid_permutation(perm):
    return (
        np.all(perm >= 0) and np.all(perm < len(perm)) and len(perm) == len(set(perm))
    )


def is_reduced_permutation(perm):
    if not is_valid_permutation(perm):
        return False
    elif len(perm) == 1:
        return True
    elif perm[0] == 0:
        return False
    elif perm[-1] == len(perm) - 1:
        return False
    else:
        return True


def normalize_permutation(perm):
    """Ensure the permutation is a numpy array and 0-indexed."""
    if not isinstance(perm, np.ndarray):
        assert isinstance(perm, list) and all(isinstance(x, int) for x in perm)
        perm = np.array(perm, dtype=int) - 1
    assert is_valid_permutation(perm)
    return perm


def reduce_permutation(perm):
    perm = normalize_permutation(perm)
    nontrivial_indices = np.where(perm != np.arange(len(perm)))[0]
    if len(nontrivial_indices) == 0:
        return np.array([0], dtype=int), None
    offset, end = nontrivial_indices.min(), nontrivial_indices.max() + 1
    reduced = perm[offset:end]
    reduced -= offset
    assert is_valid_permutation(reduced)
    return reduced, (offset, len(perm) - end)


def pad_permutation(perm, padding):
    larger_perm = np.arange(len(perm) + 2 * padding, dtype=int)
    larger_perm[padding:-padding] = perm + padding
    return larger_perm


def repr_permutation(perm):
    joiner = perm_print_joiner
    return f"[{joiner.join(map(str, perm + 1))}]"


def make_gln(target, alg=None):
    # convenience method
    if isinstance(target, list):
        return GLNHomogOp(target, alg=alg)
    elif isinstance(target, set):
        assert len(target) == 1
        l = target.pop()
        assert isinstance(l, list)
        return GLNBoostOp(l, alg=alg)
    elif isinstance(target, tuple):
        assert len(target) == 2
        l, r = target
        assert isinstance(l, list) and isinstance(r, list)
        return GLNBilocalOp(l, r, alg=alg)


class GLNHomogOp(GlobalOp):
    def __init__(self, perm, alg=None):
        self.data, *_ = reduce_permutation(perm)
        self.alg = alg

    def __repr__(self):
        return repr_permutation(self.data)

    def hardness(self):
        return 1

    def bracket_ordered(self, other):
        assert isinstance(other, GLNHomogOp)
        return GLNHomogOp.bracket_homog_homog(self, other)

    @staticmethod
    def bracket_homog_homog(left, right):
        left_perm, right_perm = left.data, right.data
        padding = len(left_perm) + 1
        right_padded = pad_permutation(right_perm, padding)
        full_size = len(right_padded)
        symmetric_group = sa.SymmetricGroup(full_size)
        sga = sa.GroupAlgebra(symmetric_group, sa.ZZ)
        # sga = sa.LieAlgebra(associative=sga)

        accum = sga.zero()
        right_padded_sga = sga((right_padded + 1).tolist())
        for i in range(full_size - len(left_perm)):
            left_extended = np.arange(full_size, dtype=int)
            left_extended[i : i + len(left_perm)] = left_perm + i
            left_extended_sga = sga((left_extended + 1).tolist())
            accum += sga.bracket(left_extended_sga, right_padded_sga)

        alg = left.alg or right.alg
        final = alg.zero()
        for sg_el, coeff in accum:
            assert coeff != 0
            perm = list(sg_el.tuple())
            perm_promoted = alg(GLNHomogOp(perm))
            final += coeff * perm_promoted

        return final

    def selfsort_tuple(self):
        return (len(self.data), self.data.tolist())

    def vacuum_ev(self):
        # Needed for the boost identification formulation of Eq. (3.20)
        return 1
        # # originally I thought this was the sign, but no, according to
        # # the footnote on page 13 it's just 1. old sign impl is below.
        # even = True
        # # brute-force n^2, could use merge sort if I really wanted to
        # for i in range(len(self.data)):
        #     for j in range(i + 1, len(self.data)):
        #         if self.data[i] > self.data[j]:
        #             even = not even
        # return 1 if even else -1


class GLNBoostOp(GlobalOp):
    def __init__(self, perm, alg=None):
        perm = normalize_permutation(perm)
        assert is_reduced_permutation(perm)
        self.data = perm
        self.alg = alg

    @staticmethod
    def reduce(perm, padding, alg):
        perm = normalize_permutation(perm)
        perm_squeezed = perm[padding:-padding] - padding
        if is_reduced_permutation(perm_squeezed):
            return alg(GLNBoostOp(perm_squeezed))
        else:
            perm_reduced, legs = reduce_permutation(perm)
            left_legs, right_legs = legs
            left_legs -= padding
            right_legs -= padding
            left_legs, right_legs = alg.base()(left_legs), alg.base()(right_legs)
            accum = alg.zero()
            accum += alg(GLNBoostOp(perm_reduced))
            if symmetric_boost_identification:
                left_legs, right_legs = (
                    (left_legs - right_legs) / 2,
                    (right_legs - left_legs) / 2,
                )
            accum -= left_legs * alg(GLNHomogOp(perm_reduced))
            accum -= right_legs * alg(GLNHomogOp([1]))
            return accum

    def __repr__(self):
        return f"B[{repr_permutation(self.data)}]"

    def hardness(self):
        return 10

    def bracket_ordered(self, other):
        assert isinstance(other, GLNHomogOp)
        return GLNBoostOp.bracket_boost_homog(self, other)

    @staticmethod
    def boost(other):
        if isinstance(other, list) or isinstance(other, np.ndarray):
            return GLNBoostOp(other)
        if isinstance(other, GLNHomogOp):
            return GLNBoostOp(other.data)
        elif other.parent() in sa.Modules:
            return extend_linear(GLNBoostOp.boost)(other)
        else:
            raise NotImplementedError()

    @staticmethod
    def bracket_boost_homog(left, right):
        left_perm, right_perm = left.data, right.data
        padding = len(right_perm) + 1  # extra padding for fun :)
        # padding = len(right_perm) - 1
        left_padded = pad_permutation(left_perm, padding)
        full_size = len(left_padded)
        symmetric_group = sa.SymmetricGroup(full_size)
        sga = sa.GroupAlgebra(symmetric_group, sa.ZZ)
        # sga = sa.LieAlgebra(associative=sga)

        accum = sga.zero()
        left_padded_sga = sga((left_padded + 1).tolist())
        for i in range(full_size - len(right_perm)):
            right_extended = np.arange(full_size, dtype=int)
            right_extended[i : i + len(right_perm)] = right_perm + i
            right_extended_sga = sga((right_extended + 1).tolist())
            comm = sga.bracket(left_padded_sga, right_extended_sga)
            accum += comm

        alg = left.alg or right.alg
        final = alg.zero()
        for sg_el, coeff in accum:
            final += coeff * GLNBoostOp.reduce(list(sg_el.tuple()), padding, alg)

        return final

    def selfsort_tuple(self):
        return (len(self.data), self.data.tolist())


class GLNBilocalOp(GlobalOp):
    def __init__(self, left, right, alg=None):
        left = normalize_permutation(left)
        assert is_reduced_permutation(left)
        right = normalize_permutation(right)
        assert is_reduced_permutation(right)
        self.data = (left, right)
        self.alg = alg

    def __repr__(self):
        left, right = self.data
        return f"[{repr_permutation(left)}|{repr_permutation(right)}]"

    def hardness(self):
        return 100

    def bracket_ordered(self, other):
        if not isinstance(other, GLNHomogOp):
            raise NotImplementedError()
        return self.bracket_bilocal_homog(self, other)

    @staticmethod
    def bracket_bilocal_homog(left, right):
        bileft, birght = left.data
        target = right.data
        full_size = max(len(bileft), len(birght), len(target))
        sg = sa.SymmetricGroup(full_size)
        sga = sa.GroupAlgebra(sg, sa.ZZ)
        bileft_sg = sga((bileft + 1).tolist())
        birght_sg = sga((birght + 1).tolist())
        target_sg = sga((target + 1).tolist())

        accum = sga.zero()

        pass

    def selfsort_tuple(self):
        return (
            len(self.data[0]),
            len(self.data[1]),
            self.data[0].tolist(),
            self.data[1].tolist(),
        )
