from ..algebra import GlobalOp
from ..tools import extend_linear

import numpy as np
import sage.all as sa


def make_gln(target):
    # for convenience
    if isinstance(target, list):
        return GLNHomogOp(target)
    elif isinstance(target, set):
        assert len(target) == 1
        l = target.pop()
        assert isinstance(l, list)
        return GLNBoostOp(GLNHomogOp(l))
    elif isinstance(target, tuple):
        assert len(target) == 2
        l, r = target
        assert isinstance(l, list) and isinstance(r, list)
        return GLNBilocalOp(GLNHomogOp(l), GLNHomogOp(r))


class GLNHomogOp(GlobalOp):
    def __init__(self, perm):
        self.data, *_ = self.reduce_permutation(perm)

    def __repr__(self):
        return f"{(self.data + 1).tolist()}".replace(" ", "")

    def hardness(self):
        return 1

    def bracket_ordered(self, other):
        assert isinstance(other, GLNHomogOp)
        return GLNHomogOp.bracket_perm_perm(self, other)

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
            perm -= 1  # lists will be 1-indexed, arrays 0-indexed
        GLNHomogOp.check_valid_permutation(perm)
        nontrivial_indices = np.where(perm != np.arange(len(perm)))[0]
        if len(nontrivial_indices) == 0:
            return np.array([0], dtype=int), None
        offset, end = nontrivial_indices.min(), nontrivial_indices.max() + 1
        reduced = perm[offset:end]
        reduced -= offset
        GLNHomogOp.check_valid_permutation(reduced)
        return reduced, (offset, len(perm) - end)

    @staticmethod
    def pad_permutation(perm, padding):
        larger_perm = np.arange(len(perm) + 2 * padding, dtype=int)
        larger_perm[padding:-padding] = perm + padding
        return larger_perm

    @staticmethod
    def bracket_perm_perm(left, right):
        left_perm, right_perm = left.data, right.data
        padding = len(left_perm) + 1
        right_padded = GLNHomogOp.pad_permutation(right_perm, padding)
        full_size = len(right_padded)
        symmetric_group = sa.SymmetricGroup(full_size)
        sga = sa.GroupAlgebra(symmetric_group, sa.ZZ)
        sga = sa.LieAlgebra(associative=sga)

        accum = sga.zero()
        right_padded_sga = sga((right_padded + 1).tolist())
        for i in range(full_size - len(left_perm)):
            left_extended = np.arange(full_size, dtype=int)
            left_extended[i : i + len(left_perm)] = left_perm + i
            left_extended_sga = sga((left_extended + 1).tolist())
            accum += left_extended_sga.bracket(right_padded_sga)

        final_dict = {}
        for sg_el, coeff in accum:
            assert coeff != 0
            perm = list(sg_el.tuple())
            perm_promoted = GLNHomogOp(perm)
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


class GLNBoostOp(GlobalOp):
    def __init__(self, child):
        assert isinstance(child, GLNHomogOp)
        self.data = child

    def __repr__(self):
        return f"B[{self.data}]"

    def hardness(self):
        return 10

    def bracket_ordered(self, other):
        assert isinstance(other, GLNHomogOp)
        return GLNBoostOp.bracket_boost_perm(self, other)

    @staticmethod
    def boost(other):
        if isinstance(other, GLNHomogOp):
            return GLNBoostOp(other)
        elif other.parent() in sa.LieAlgebras:
            return extend_linear(GLNBoostOp)(other)
        else:
            raise NotImplementedError()

    @staticmethod
    def bracket_boost_perm(left, right):
        Kpoly = sa.PolynomialRing(sa.ZZ, "k")
        origin_offset = Kpoly.gen()
        left_perm, right_perm = left.data.data, right.data
        padding = len(right_perm) + 1  # extra padding for fun :)
        # padding = len(right_perm) - 1
        left_padded = GLNHomogOp.pad_permutation(left_perm, padding)
        full_size = len(left_padded)
        symmetric_group = sa.SymmetricGroup(full_size)
        sga = sa.GroupAlgebra(symmetric_group, Kpoly)
        sga = sa.LieAlgebra(associative=sga)

        accum = sga.zero()
        left_padded_sga = sga((left_padded + 1).tolist())
        for i in range(full_size - len(right_perm)):
            right_extended = np.arange(full_size, dtype=int)
            right_extended[i : i + len(right_perm)] = right_perm + i
            right_extended_sga = sga((right_extended + 1).tolist())
            comm = left_padded_sga.bracket(right_extended_sga)
            accum += (origin_offset + padding) * comm
            # we could have done left_legs -= padding later instead of doing
            # origin_offset + padding here, but this feels more morally correct
            # accum += origin_offset * comm  # less morally correct, but works

        final_dict = {}
        identity_multiple = Kpoly.zero()
        for sg_el, coeff in accum:
            perm = list(sg_el.tuple())
            perm_reduced, legs = GLNHomogOp.reduce_permutation(perm)
            left_legs, right_legs = legs
            perm_promoted = GLNHomogOp(perm_reduced)
            pre_coeff = final_dict.get(perm_promoted, 0)

            # left_legs -= padding # incompatible with + padding on accum
            right_legs -= padding  # not necessary, but morally correct

            # this perm_offset chicanery is an idea to do a simpler thing than
            # the complicated identification relations (3.20) in the paper, and
            # empirically it's worked so far. the idea is that we essentially
            # force every single permutation in the boost (i.e. which has a
            # linear coefficient of origin_offset) to be shifted by the actual
            # offset of the permutation; this should be intuitively clear.
            # similarly, one can view this as implementing the spectator leg
            # relations in (3.19) directly.
            final_dict[perm_promoted] = pre_coeff + coeff.subs(
                origin_offset - left_legs
            )
            # we need to subtract the corresponding factors of the identity to
            # compensate for the right-side spectator legs; again, this can be
            # viewed as implementing (3.19) directly.
            identity_multiple -= (
                perm_promoted.vacuum_ev() * right_legs * coeff.coefficient(1)
            )

        final_dict[GLNHomogOp([1])] = identity_multiple

        return final_dict

    @staticmethod
    def bracket_boost_perm_OLDMETHOD(left, right):
        Kpoly = sa.PolynomialRing(sa.ZZ, "k")
        origin_offset = Kpoly.gen()
        left_perm, right_perm = left.data.data, right.data
        padding = len(left_perm) + 1
        right_padded = GLNHomogOp.pad_permutation(right_perm, padding)
        full_size = len(right_padded)
        symmetric_group = sa.SymmetricGroup(full_size)
        sga = sa.GroupAlgebra(symmetric_group, Kpoly)
        sga = sa.LieAlgebra(associative=sga)

        accum = sga.zero()
        right_padded_sga = sga((right_padded + 1).tolist())
        for i in range(full_size - len(left_perm)):
            left_extended = np.arange(full_size, dtype=int)
            left_extended[i : i + len(left_perm)] = left_perm + i
            left_extended_sga = sga((left_extended + 1).tolist())
            comm = left_extended_sga.bracket(right_padded_sga)
            # # this max() bit is really subtle: it has to do with the fact that
            # # if the left permutation starts before the right (which we've
            # # localized), we aren't actually acting at the site of that
            # # localized right permutation, so we need to think about which
            # # site we're actually acting on and use the appropriate shift.
            # # also, since the origin offset is arbitrary, I think we
            # # could have done max(i, padding) instead, but we're doing it this
            # # way to more directly reflect the choice of spin chain origin.
            # accum += (origin_offset + max(i - padding, 0)) * comm
            # ACTUALLY, we'll do something else instead; see the comment in
            # the final loop for more details.
            accum += (origin_offset + i - padding) * comm
            # it seems that subtracting padding does nothing, but it's moral

        final_dict = {}
        identity_multiple = Kpoly.zero()
        for sg_el, coeff in accum:
            perm = list(sg_el.tuple())
            perm_reduced, legs = GLNHomogOp.reduce_permutation(perm)
            left_legs, right_legs = legs
            perm_promoted = GLNHomogOp(perm_reduced)
            pre_coeff = final_dict.get(perm_promoted, 0)

            # the following seem to be unnecessary but are morally correct
            left_legs -= padding
            right_legs -= padding

            # this perm_offset chicanery is very very subtle: I think that it's
            # a valid alternative to combining the max(i-padding, 0) idea from
            # above with the identification relations (3.20) in the paper, and
            # empirically it's worked so far. the idea is that we essentially
            # force every single permutation in the boost (i.e. which has a
            # linear coefficient of origin_offset) to be shifted by the actual
            # offset of the permutation; this should be intuitively clear.
            # similarly, one can view this as implementing the spectator leg
            # relations in (3.19) directly.
            final_dict[perm_promoted] = pre_coeff + coeff.subs(
                origin_offset - left_legs
            )
            # we need to subtract the corresponding factors of the identity to
            # compensate for the right-side spectator legs; again, this can be
            # viewed as implementing (3.19) directly.
            identity_multiple -= (
                perm_promoted.vacuum_ev() * right_legs * coeff.coefficient(1)
            )

        final_dict[GLNHomogOp([1])] = identity_multiple

        return final_dict

    def selfsort_tuple(self):
        return (*self.data.sort_order(),)


class GLNBilocalOp(GlobalOp):
    def __init__(self, left, right):
        assert isinstance(left, GLNHomogOp) and isinstance(right, GLNHomogOp)
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
