import stat
from functools import cache
from operator import le

import numpy as np
import sage.all as sa

from ..algebra import GlobalOp
from ..config import (
    incl_antiwrap_in_bilocal_bracket,
    perm_print_joiner,
    symmetric_boost_identification,
    use_numba,
)
from ..tools import (
    AccumWrapper,
    extend_linear,
    map_collect_elements,
    map_elements,
    sum_dicts,
)

try:
    from line_profiler import profile
except ImportError:
    profile = lambda *_, **__: lambda fn: fn

if use_numba:
    from numba import njit, prange
else:
    njit = lambda *_, **__: lambda fn: fn
    prange = range


@njit(cache=True)
def is_valid_permutation(perm):
    if not np.all(perm >= 0):
        return False
    if not np.all(perm < len(perm)):
        return False
    seen = np.zeros(len(perm), dtype=np.bool_)
    seen[perm] = True
    return np.all(seen)


@njit(cache=True)
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
    if type(perm) is not np.ndarray:
        if isinstance(perm, sa.SageObject):
            perm = list(perm.tuple())
        assert type(perm) is list  # and all(isinstance(x, int) for x in perm)
        perm = np.array(perm, dtype=np.int32) - 1
    return perm


@njit(cache=True)
def reduce_permutation(perm, padding=0):
    nontrivial_indices = perm != np.arange(len(perm))
    if not np.any(nontrivial_indices):
        return np.array([0], dtype=np.int32), 0, len(perm) - 1 - 2 * padding
    offset, end = 0, len(perm)
    while not nontrivial_indices[offset]:
        offset += 1
    while not nontrivial_indices[end - 1]:
        end -= 1
    offset, end = int(offset), int(end)  # avoids sage weirdness with np.int64
    reduced = perm[offset:end]
    reduced = reduced.astype(np.int32)  # copies as well
    reduced -= offset
    # assert is_valid_permutation(reduced)
    return reduced, offset - padding, len(perm) - end - padding


@njit(cache=True)
def pad_permutation(perm, padding):
    larger_perm = np.arange(len(perm) + 2 * padding, dtype=np.int32)
    larger_perm[padding : len(larger_perm) - padding] = perm + padding
    return larger_perm


def repr_permutation(perm):
    joiner = perm_print_joiner
    return f"[{joiner.join(map(str, perm + 1))}]"


def make_gln(target, alg=None):
    # convenience method
    if type(target) is list or type(target) is np.ndarray:
        return GLNHomogOp(target, alg=alg)
    elif type(target) is tuple:
        if len(target) == 1:
            (l,) = target
            return GLNBoostOp(l, alg=alg)
        elif len(target) == 2:
            l, r = target
            return GLNBilocalOp(l, r, alg=alg)
        else:
            raise ValueError(f"Invalid target {target} for make_gln")
    else:
        raise TypeError(
            f"Invalid type {type(target)} for make_gln, expected list or tuple"
        )


@njit(cache=True)
def _drag_right_on_left(left_perm, right_perm):
    # padding = len(right_perm) + 1  # extra padding for fun :)
    padding = len(right_perm)
    # padding = len(right_perm) - 1
    left_padded = pad_permutation(left_perm, padding)
    full_size = len(left_padded)
    num = full_size - len(right_perm)
    pos_accum = np.zeros((num, full_size), dtype=np.int32)
    neg_accum = np.zeros((num, full_size), dtype=np.int32)
    for i in range(num):
        right_extended = np.arange(full_size, dtype=np.int32)
        right_extended[i : i + len(right_perm)] = right_perm + i
        lr, rl = left_padded[right_extended], right_extended[left_padded]
        pos_accum[i] = lr
        neg_accum[i] = rl

    pos_flags = np.ones(len(pos_accum), dtype=np.bool_)
    neg_flags = np.ones(len(neg_accum), dtype=np.bool_)
    for i in range(len(pos_accum)):
        for j in range(len(neg_accum)):
            if neg_flags[j] and np.array_equal(pos_accum[i], neg_accum[j]):
                pos_flags[i] = False
                neg_flags[j] = False
                break

    return pos_accum, neg_accum, pos_flags, neg_flags, padding


def drag_right_on_left(left_perm, right_perm, one=1):
    neg_one = -one
    pos, neg, pos_flags, neg_flags, padding = _drag_right_on_left(left_perm, right_perm)
    accum = []
    num = len(pos_flags)
    for i in range(num):
        if pos_flags[i]:
            accum.append((pos[i], one))
        if neg_flags[i]:
            accum.append((neg[i], neg_one))

    return accum, padding


@njit(cache=True)
def perm_compose_sided(left_perm, right_perm):
    # assert is_valid_permutation(left_perm)
    # assert is_valid_permutation(right_perm)
    # sg = sa.SymmetricGroup(max(len(left_perm), len(right_perm)))
    # left, right = sg((left_perm + 1).tolist()), sg((right_perm + 1).tolist())
    # lr = list((left * right).tuple())
    # rl = list((right * left).tuple())
    max_len = max(len(left_perm), len(right_perm))
    identity = np.arange(max_len, dtype=np.int32)
    left, right = identity.copy(), identity.copy()
    left[: len(left_perm)] = left_perm
    right[: len(right_perm)] = right_perm
    lr = left[right]
    rl = right[left]
    return lr, rl


@njit(cache=True)
def pair_pad(a, a_pad, b, b_pad):
    a_pad = a_pad - b_pad
    if a_pad > 0:
        return pad_permutation(a, a_pad), b
    elif a_pad < 0:
        return a, pad_permutation(b, -a_pad)
    else:
        return a, b


class GLNHomogOp(GlobalOp):
    @profile
    def __init__(self, perm, alg=None):
        if type(perm) is not np.ndarray:
            perm = normalize_permutation(perm)
        # assert is_valid_permutation(perm)
        self.data, _, _ = reduce_permutation(perm)
        self.alg = alg

    def __repr__(self):
        return repr_permutation(self.data)

    def hardness(self):
        return 1

    def bracket_ordered(self, other):
        assert type(other) is GLNHomogOp
        return GLNHomogOp.bracket_homog_homog(self, other)

    def selfsort_tuple(self):
        return (len(self.data), self.data.tobytes())

    @staticmethod
    @profile
    def bracket_homog_homog(left, right):
        alg = left.alg or right.alg
        ring = alg.base()

        left_perm, right_perm = left.data, right.data
        dragged, _ = drag_right_on_left(left_perm, right_perm, ring.one())

        final = map_collect_elements(
            dragged, lambda k, v: (GLNHomogOp(k, alg=alg), v), ring.zero()
        )

        # final = alg(final)
        return final

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


class GLNBoostOp(GlobalOp):  # XXX THIS CLASS IS DEPRECATED
    r"""
    \begin{align}
    \mc B[A]&\equiv \sum_a aA_a\\
    &= \sum_{0<a\leq N-|A|} aA_a\\
    \mc B[\leg A]&=\mc B[A]-[A]\\
    \mc B[A \leg]&=\mc B[A]-[1]\ev{A}
    {}\end{align}
    """

    def __init__(self, perm, alg=None):
        # raise DeprecationWarning(
        #     """
        #     Deprecated because there seem to exist some currently-unknown relations
        #     between boost operators this causes inhomogeneities that are resolved by
        #     using bilocal-boosts, which are more well-behaved. Figuring out these
        #     relations would be an interesting topic of exploration, and would probably
        #     only take a few days.
        #     """
        # )
        perm = normalize_permutation(perm)
        assert is_reduced_permutation(perm)
        self.data = perm
        self.alg = alg

    def __repr__(self):
        return f"B[{repr_permutation(self.data)}]"

    def hardness(self):
        return 10

    def bracket_ordered(self, other):
        assert type(other) is GLNHomogOp
        return GLNBoostOp.bracket_boost_homog(self, other)

    def selfsort_tuple(self):
        return (len(self.data), self.data.tobytes())

    @staticmethod
    def boost(other):
        if type(other) is list or type(other) is np.ndarray:
            return GLNBoostOp(other)
        if type(other) is GLNHomogOp:
            return GLNBoostOp(other.data, other.alg)
        elif other.parent() in sa.Modules:
            return extend_linear(GLNBoostOp.boost)(other)
        else:
            raise NotImplementedError()

    @staticmethod
    def reduce(perm, padding, alg):
        # TODO check if it's possible to reduce boosts of disjoint permutations
        ring = alg.base()
        perm = normalize_permutation(perm)
        perm_squeezed = perm[padding : len(perm) - padding] - padding
        if is_reduced_permutation(perm_squeezed):
            return (GLNBoostOp(perm_squeezed), 1)
        else:
            perm_reduced, left_legs, right_legs = reduce_permutation(perm, padding)
            left_legs, right_legs = ring(left_legs), ring(right_legs)
            if symmetric_boost_identification:
                # the asymmetric identification is, IMO, much more intuitive,
                # but this is the one that matches eq. (3.20) in the paper
                left_legs, right_legs = (
                    (left_legs - right_legs) / 2,
                    (right_legs - left_legs) / 2,
                )
            accum_list = [
                (GLNBoostOp(perm_reduced, alg=alg), 1),
                (GLNHomogOp(perm_reduced, alg=alg), -left_legs),
                (GLNHomogOp([1], alg=alg), -right_legs),
            ]
            return accum_list

    @staticmethod
    def bracket_boost_homog(left, right):
        # the methodology here is to write sum_a a sum_b [l(a), r(b)] and so
        # we're just expanding the inner sum and then calling that a boost op
        # (where we then need to do the appropriate operator identification)
        alg = left.alg or right.alg
        ring = alg.base()

        left_perm, right_perm = left.data, right.data
        dragged, padding = drag_right_on_left(left_perm, right_perm, ring.one())

        final_list = []
        for sg_el, coeff in dragged:
            coeff = ring(coeff)  # probably already true
            final_list.extend(
                [(k, v * coeff) for (k, v) in GLNBoostOp.reduce(sg_el, padding, alg)]
            )
        final = map_collect_elements(final_list, lambda k, v: (k, v), ring.zero())

        # return alg(final) if final else alg.zero()
        return final


class GLNBilocalOp(GlobalOp):
    r"""
    This is my own version of the bilocal operator; it's different
    from the original paper's and less symmetric, but IMO easier to
    work with. The definition is:
    \begin{align}
    [A/B]&\equiv\frac12\sum_a\sum_{b>a}\Bqty{A_a, B_b}\\
    \antiwrap{A,B}&\equiv\frac12\sum_a\Bqty{A_a,B_a}\\
    [A|B]&\equiv [A/B]+\frac12\antiwrap{A, B}\\
    {}\end{align}
    See the documentation of `reduce_slashed` for the identifications.
    """

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
        if type(other) is not GLNHomogOp:
            raise NotImplementedError()
        return self.bracket_bilocal_homog(self, other)

    def selfsort_tuple(self):
        return (
            len(self.data[0]),
            len(self.data[1]),
            self.data[0].tobytes(),
            self.data[1].tobytes(),
        )

    @staticmethod
    def bilocalize(left, right):
        if type(left) is list or type(left) is np.ndarray:
            assert type(right) is list or type(right) is np.ndarray
            return GLNBilocalOp(left, right)
        elif type(left) is GLNHomogOp:
            assert type(right) is GLNHomogOp
            alg = left.alg or right.alg
            assert alg is not None, "Algebra must be provided for bilocalization"
            return GLNBilocalOp(left.data, right.data, alg=alg)
        elif left.parent() in sa.Modules:
            assert right.parent() in sa.Modules
            left_len = 0
            for l, _ in left:
                left_len = max(left_len, len(l.data))
            right_len = 0
            for r, _ in right:
                right_len = max(right_len, len(r.data))

            alg = left.parent()

            accum_list = []
            for l, lc in left:
                for r, rc in right:
                    assert type(l) is GLNHomogOp and type(r) is GLNHomogOp
                    # lx = np.arange(left_len, dtype=int)
                    # rx = np.arange(right_len, dtype=int)
                    # if len(l.data):
                    #     lx[: len(l.data)] = l.data
                    # if len(r.data):
                    #     rx[: len(r.data)] = r.data
                    # reduced_slashed_list = GLNBilocalOp.reduce_slashed(
                    #     lx, 0, rx, 0, alg
                    # )[0]
                    # reduced_slashed = map_collect_elements(
                    #     reduced_slashed_list,
                    #     lambda k, v: (k, v),
                    #     alg.base().zero(),
                    # )
                    # accum += lc * rc * alg(reduced_slashed)
                    bl = GLNBilocalOp(l.data, r.data, alg=alg)
                    accum_list.append((bl, lc * rc))
            accum = map_collect_elements(
                accum_list,
                lambda k, v: (k, v),
                alg.base().zero(),
            )
            return alg(accum)

        else:
            raise NotImplementedError("Bilocalization for " + type(left))

    @staticmethod
    @profile
    def reduce_slashed(left, lpad, rght, rpad, append_antiwrap):
        r"""
        \begin{align}
        [A/B]&=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|} \Bqty{A_a, B_b}\\
        [A/\leg B]&=\frac12\sum_{0<a}\sum_{a<b\leq N-|\leg B|} \Bqty{A_a, \leg B_b}\\
        &=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|-1} \Bqty{A_a,  B_{b+1}}\\
        &=\frac12\sum_{0<a}\sum_{a+1<b\leq N-|B|} \Bqty{A_a,  B_{b}}\\
        &=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|} \Bqty{A_a,  B_{b}}-\frac12\sum_{a}\Bqty{A_a,  B_{a+1}}\\
        &=[A/B]-\antiwrap{A, \leg B}\\
        [A/B\leg ]&=\frac12\sum_{0<a}\sum_{a<b\leq N-|B\leg |} \Bqty{A_a, B_b\leg }\\
        &=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|-1} \Bqty{A_a,  B_{b}}\\
        &=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|} \Bqty{A_a,  B_{b}}-{ \frac12\sum_{a}\Bqty{A_a,  B_{N-|B|-1}} }\\
        &=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|} \Bqty{A_a,  B_{b}}-{ \sum_a A_a }\\
        &=[A/B]-[A]\\
        [\leg A/B]&=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|} \Bqty{\leg A_a, B_b\leg }\\
        &=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|} \Bqty{A_{a+1},  B_{b}}\\
        &=\frac12\sum_{1<a}\sum_{a-1<b\leq N-|B|} \Bqty{A_{a},  B_{b}}\\
        &=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|} \Bqty{A_a,  B_{b}}- \frac12\sum_{b}\Bqty{A_1,  B_{b}}+\frac12\sum_{a}\Bqty{A_{a},  B_{a}} \\
        &=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|} \Bqty{A_a,  B_{b}}- \sum_{b}B_b+\frac12\sum_{a}\Bqty{A_{a},  B_{a}} \\
        &=[A/B]-[B]+\antiwrap{A,B}\\
        [A\leg /B]&=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|} \Bqty{ A_a\leg, B_b\leg }\\
        &=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|} \Bqty{A_{a},  B_{b}}\\
        &=[A/B]\\
        {}\end{align}
        """
        left = normalize_permutation(left)
        rght = normalize_permutation(rght)
        left_red, ll_legs, lr_legs = reduce_permutation(left, padding=lpad)
        rght_red, rl_legs, rr_legs = reduce_permutation(rght, padding=rpad)

        left_hom, rght_hom = GLNHomogOp(left_red), GLNHomogOp(rght_red)
        primary = GLNBilocalOp(left_red, rght_red)

        accum = []
        # accum = AccumWrapper(accum)
        # accum += alg(primary)
        # accum -= ll_legs * left_hom.vacuum_ev() * alg(rght_hom)
        # accum -= rr_legs * rght_hom.vacuum_ev() * alg(left_hom)
        accum.append((primary, 1))
        if ll_legs != 0:
            accum.append((rght_hom, -ll_legs * left_hom.vacuum_ev()))
        if rr_legs != 0:
            accum.append((left_hom, -rr_legs * rght_hom.vacuum_ev()))
        # the below updates accum
        GLNBilocalOp.reduce_slashed_recurse(
            left_red, rght_red, ll_legs, rl_legs, accum, append_antiwrap
        )
        # note that lr isn't used because I think it's only on boundary.

        accum = accum._wrapped if type(accum) is AccumWrapper else accum

        # accum = map_collect_elements(accum, lambda k, v: (alg(k), v), alg.base().zero())

        return accum, primary

    @staticmethod
    @profile
    def reduce_slashed_recurse(l_red, r_red, ll_legs, rl_legs, accum, append_antiwrap):
        if ll_legs > 0:
            # comms = perm_compose_sided(l_red, r_red)
            # accum += GLNBilocalOp.antiwrap(comms, alg)
            l, r = pair_pad(l_red, ll_legs - 1, r_red, rl_legs)
            append_antiwrap(l, r, 1, accum.append)
            return GLNBilocalOp.reduce_slashed_recurse(
                l_red, r_red, ll_legs - 1, rl_legs, accum, append_antiwrap
            )
        elif ll_legs < 0:
            # comms = perm_compose_sided(l_red, pad_permutation(r_red, -ll_legs))
            # accum -= GLNBilocalOp.antiwrap(comms, alg)
            l, r = pair_pad(l_red, ll_legs, r_red, rl_legs)
            append_antiwrap(l, r, -1, accum.append)
            return GLNBilocalOp.reduce_slashed_recurse(
                l_red, r_red, ll_legs + 1, rl_legs, accum, append_antiwrap
            )
        elif rl_legs > 0:
            # comms = perm_compose_sided(l_red, pad_permutation(r_red, rl_legs))
            # accum -= GLNBilocalOp.antiwrap(comms, alg)
            l, r = pair_pad(l_red, 0, r_red, rl_legs)
            append_antiwrap(l, r, -1, accum.append)
            return GLNBilocalOp.reduce_slashed_recurse(
                l_red, r_red, ll_legs, rl_legs - 1, accum, append_antiwrap
            )
        elif rl_legs < 0:
            # comms = perm_compose_sided(l_red, r_red)
            # accum += GLNBilocalOp.antiwrap(comms, alg)
            l, r = pair_pad(l_red, 0, r_red, rl_legs + 1)
            append_antiwrap(l, r, 1, accum.append)
            return GLNBilocalOp.reduce_slashed_recurse(
                l_red, r_red, ll_legs, rl_legs + 1, accum, append_antiwrap
            )
        else:
            return accum

    @staticmethod
    @profile
    def _append_antiwrap_proto(l, r, coeff, op, alg, half):
        comms = perm_compose_sided(l, r)
        for ix in [0, 1]:
            cur = GLNHomogOp(comms[ix], alg=alg)
            half_coeff = half * coeff
            op((cur, half_coeff))

    @staticmethod
    @profile
    def bracket_bilocal_homog(left, right):
        # it's possible that all the antiwrap terms will in general cancel out
        # in this method, but I haven't proved that it's true and it seems easy
        # to make a mistake while doing so.
        bileft, birght = left.data
        target = right.data
        # for short, these are bl, br, tg.

        alg = left.alg or right.alg
        ring = alg.base()
        half = ring.one() / 2
        accum_slashed = []
        accum_antiwrap = []

        def append_antiwrap(left, right, coeff, op):
            GLNBilocalOp._append_antiwrap_proto(left, right, coeff, op, alg, half)

        br_tg_drag, br_tg_pad = drag_right_on_left(birght, target, ring.one())
        for br_tg_perm, coeff in br_tg_drag:
            coeff = ring(coeff)
            reduced, primary = GLNBilocalOp.reduce_slashed(
                bileft, 0, br_tg_perm, br_tg_pad, append_antiwrap
            )
            # accum_slashed += coeff * reduced
            accum_slashed.extend([(k, v * coeff) for (k, v) in reduced])

            red_left, red_rght = primary.data
            # red_comms = perm_compose_sided(red_left, red_rght)
            # accum_antiwrap -= coeff * GLNBilocalOp.antiwrap(red_comms, alg) / 2
            append_antiwrap(red_left, red_rght, -coeff / 2, accum_antiwrap.append)

        bl_tg_drag, bl_tg_pad = drag_right_on_left(bileft, target, ring.one())
        for bl_tg_perm, coeff in bl_tg_drag:
            coeff = ring(coeff)
            reduced, primary = GLNBilocalOp.reduce_slashed(
                bl_tg_perm, bl_tg_pad, birght, 0, append_antiwrap
            )
            # accum_slashed += coeff * reduced
            accum_slashed.extend([(k, v * coeff) for (k, v) in reduced])

            red_left, red_rght = primary.data
            # red_comms = perm_compose_sided(red_left, red_rght)
            # accum_antiwrap -= coeff * GLNBilocalOp.antiwrap(red_comms, alg) / 2
            append_antiwrap(red_left, red_rght, -coeff / 2, accum_antiwrap.append)

        bl_br_comms = perm_compose_sided(bileft, birght)
        # accum_antiwrap += (
        #     GLNBilocalOp.antiwrap(bl_br_comms, alg).bracket(alg(right)) / 2
        # )
        quarter = ring.one() / 4
        for ix in [0, 1]:
            cur = GLNHomogOp(bl_br_comms[ix], alg=alg)
            cur = cur.bracket(right)
            map_elements(
                cur,
                accum_antiwrap.append,
                lambda k, v: (k, v * quarter),
            )

        accum_sl = map_collect_elements(
            accum_slashed,
            lambda k, v: (k, v),
            alg.base().zero(),
        )

        if incl_antiwrap_in_bilocal_bracket:
            accum_aw = map_collect_elements(
                accum_antiwrap,
                lambda k, v: (k, v),
                alg.base().zero(),
            )
            return sum_dicts(accum_sl, accum_aw, zero=alg.base().zero())
        else:
            return accum_sl
