from abc import ABC, abstractmethod
import numpy as np
import sage.all
from sage.algebras.group_algebra import GroupAlgebra
from sage.combinat.all import CombinatorialFreeModule
from sage.groups.perm_gps.all import SymmetricGroup
from sage.rings.polynomial.all import PolynomialRing


def extend_linear(fn):
    return (
        lambda arg: fn(arg)
        if not hasattr(arg, "__iter__")
        else sum(coeff * arg.parent()(fn(elem)) for elem, coeff in arg)
    )


def extend_bilinear(fn):
    xtl = extend_linear
    return lambda left, right: xtl(lambda L: xtl(lambda R: fn(L, R))(right))(left)


class GlobalOp(ABC):
    def __hash__(self):
        if isinstance(self.data, np.ndarray):
            return hash((tuple(self.data), type(self)))
        return hash((self.data, type(self)))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if isinstance(self.data, np.ndarray) and isinstance(other.data, np.ndarray):
            return np.array_equal(self.data, other.data)
        elif isinstance(self.data, np.ndarray) or isinstance(other.data, np.ndarray):
            return False
        return self.data == other.data

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def hardness(self):
        pass

    @abstractmethod
    def bracket_ordered(self, other):
        pass

    def _bracket_(self, other):
        assert isinstance(other, GlobalOp)
        if other.hardness() > self.hardness():
            return {k: -v for k, v in other.bracket_ordered(self).items()}
        else:
            return self.bracket_ordered(other)

    def sort_order(self):
        return (-self.hardness(), *self.selfsort_tuple())

    @abstractmethod
    def selfsort_tuple(self):
        pass


class GlobalLengthOp(GlobalOp):
    def __init__(self):
        self.data = None

    def __repr__(self):
        return "N"

    def hardness(self):
        return 15

    def bracket_ordered(self, other):
        raise NotImplementedError()

    def selfsort_tuple(self):
        return tuple()


class GlobalPermOp(GlobalOp):
    def __init__(self, perm):
        self.data, _ = GlobalPermOp.reduce_permutation(perm)

    def __repr__(self):
        return f"{(self.data + 1).tolist()}".replace(" ", "")

    def hardness(self):
        return 1

    def bracket_ordered(self, other):
        assert isinstance(other, GlobalPermOp)
        return GlobalPermOp.bracket_perm_perm(self, other)

    @staticmethod
    def check_valid_permutation(perm):
        assert np.all(perm >= 0)
        assert np.all(perm < len(perm))
        assert len(perm) == len(set(perm))

    @staticmethod
    def reduce_permutation(perm):
        if not isinstance(perm, np.ndarray):
            assert isinstance(perm, list) and all(isinstance(x, int) for x in perm)
            perm = np.array(perm, dtype=int)
            perm -= 1  # Convert to zero-based indexing
        GlobalPermOp.check_valid_permutation(perm)
        nontrivial_indices = np.where(perm != np.arange(len(perm)))[0]
        if len(nontrivial_indices) == 0:
            return np.array([0], dtype=int), None
        offset, end = nontrivial_indices.min(), nontrivial_indices.max()
        perm = perm[offset : end + 1]
        perm -= offset
        GlobalPermOp.check_valid_permutation(perm)
        return perm, offset

    @staticmethod
    def pad_permutation(perm, padding):
        larger_perm = np.arange(len(perm) + 2 * padding, dtype=int)
        larger_perm[padding:-padding] = perm + padding
        return larger_perm

    @staticmethod
    def bracket_perm_perm(left, right):
        left_perm, right_perm = left.data, right.data
        padding = len(left_perm) + 1
        right_padded = GlobalPermOp.pad_permutation(right_perm, padding)
        full_size = len(right_padded)
        symmetric_group = SymmetricGroup(full_size)
        sga = GroupAlgebra(symmetric_group, sage.all.ZZ)

        accum = sga.zero()
        right_padded_sga = sga((right_padded + 1).tolist())
        for i in range(full_size - len(left_perm)):
            left_extended = np.arange(full_size, dtype=int)
            left_extended[i : i + len(left_perm)] = left_perm + i
            left_extended_sga = sga((left_extended + 1).tolist())
            lr = left_extended_sga * right_padded_sga
            rl = right_padded_sga * left_extended_sga
            accum += lr - rl

        final_dict = {}
        for sg_el, coeff in accum:
            assert coeff != 0
            perm = list(sg_el.tuple())
            perm_promoted = GlobalPermOp(perm)
            pre_coeff = final_dict.get(perm_promoted, 0)
            final_dict[perm_promoted] = pre_coeff + coeff

        return final_dict

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


class GlobalBoostOp(GlobalOp):
    def __init__(self, child):
        assert isinstance(child, GlobalPermOp)
        self.data = child

    def __repr__(self):
        return f"B[{self.data}]"

    def hardness(self):
        return 10

    def bracket_ordered(self, other):
        assert isinstance(other, GlobalPermOp)
        return GlobalBoostOp.bracket_boost_perm(self, other)

    @staticmethod
    def boost(other):
        if isinstance(other, GlobalPermOp):
            return GlobalBoostOp(other)
        elif isinstance(other.parent(), CombinatorialFreeModule):
            return extend_linear(GlobalBoostOp)(other)
        else:
            raise NotImplementedError()

    @staticmethod
    def bracket_boost_perm(left, right):
        Kpoly = PolynomialRing(sage.all.ZZ, "k")
        origin_offset = Kpoly.gen()
        left_perm, right_perm = left.data.data, right.data
        padding = len(left_perm) + 1
        right_padded = GlobalPermOp.pad_permutation(right_perm, padding)
        full_size = len(right_padded)
        symmetric_group = SymmetricGroup(full_size)
        sga = GroupAlgebra(symmetric_group, Kpoly)

        accum = sga.zero()
        right_padded_sga = sga((right_padded + 1).tolist())
        for i in range(full_size - len(left_perm)):
            left_extended = np.arange(full_size, dtype=int)
            left_extended[i : i + len(left_perm)] = left_perm + i
            left_extended_sga = sga((left_extended + 1).tolist())
            lr = left_extended_sga * right_padded_sga
            rl = right_padded_sga * left_extended_sga
            # # this max() bit is really subtle: it has to do with the fact that
            # # if the left permutation starts before the right (which we've
            # # localized), we aren't actually acting at the site of that
            # # localized right permutation, so we need to think about which
            # # site we're actually acting on and use the appropriate shift.
            # # also, since the origin offset is arbitrary, I think we
            # # could have done max(i, padding) instead, but we're doing it this
            # # way to more directly reflect the choice of spin chain origin.
            # accum += (origin_offset + max(i - padding, 0)) * (lr - rl)
            # ACTUALLY, we'll do something else instead; see the comment in
            # the final loop for more details.
            accum += (origin_offset + i) * (lr - rl)

        final_dict = {}
        identity_multiple = 0
        for sg_el, coeff in accum:
            perm = list(sg_el.tuple())
            perm_reduced, perm_offset = GlobalPermOp.reduce_permutation(perm)
            perm_promoted = GlobalPermOp(perm_reduced)
            pre_coeff = final_dict.get(perm_promoted, 0)

            # this perm_offset chicanery is very very subtle: I think that it's
            # a valid alternative to combining the max(i - padding) idea from
            # above with the identification relations (3.20) in the paper, and
            # empirically it's worked so far. the idea is that we essentially
            # force every single permutation in the boost (i.e. which has a
            # linear coefficient of origin_offset) to be shifted by the actual
            # offset of the permutation; this should be intuitively clear.
            final_dict[perm_promoted] = pre_coeff + coeff.subs(
                origin_offset - perm_offset
            )
            # we need to subtract the corresponding factors of the identity to
            # reproduce the results of the paper a la Eq. (3.20).
            identity_multiple += (
                perm_promoted.vacuum_ev() * perm_offset * coeff.coefficient(1)
            )

        final_dict[GlobalPermOp([1])] = identity_multiple

        return final_dict

    def selfsort_tuple(self):
        return (*self.data.sort_order(),)


class GlobalBilocalOp(GlobalOp):
    def __init__(self, left, right):
        assert isinstance(left, GlobalPermOp)
        assert isinstance(right, GlobalPermOp)
        self.data = (left, right)

    def __repr__(self):
        left, right = self.data
        return f"[ {left} | {right} ]"

    def hardness(self):
        return 100

    def bracket_ordered(self, other):
        raise NotImplementedError()

    def selfsort_tuple(self):
        return (*self.data[0].sort_order(), *self.data[1].sort_order())


def operators_ordered(first, second):
    op_order = [GlobalBilocalOp, GlobalBoostOp, GlobalLengthOp, GlobalPermOp]
    return op_order.index(type(first)) <= op_order.index(type(second))
