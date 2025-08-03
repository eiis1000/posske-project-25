from ..algebra import GlobalOp
from ..config import (
    perm_print_joiner,
    symmetric_boost_identification,
    incl_antiwrap_in_bilocal_bracket,
)
from ..tools import extend_linear, AccumWrapper

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


def reduce_permutation(perm, padding=0):
    perm = normalize_permutation(perm)
    nontrivial_indices = np.where(perm != np.arange(len(perm)))[0]
    if len(nontrivial_indices) == 0:
        return np.array([0], dtype=int), (0, len(perm) - 1 - 2 * padding)
    offset, end = nontrivial_indices.min(), nontrivial_indices.max() + 1
    offset, end = int(offset), int(end)  # avoids sage weirdness with np.int64
    reduced = perm[offset:end]
    reduced = reduced - offset
    assert is_valid_permutation(reduced)
    return reduced, (offset - padding, len(perm) - end - padding)


def pad_permutation(perm, padding):
    larger_perm = np.arange(len(perm) + 2 * padding, dtype=int)
    larger_perm[padding : len(larger_perm) - padding] = perm + padding
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


def drag_right_on_left(left_perm, right_perm):
    padding = len(right_perm) + 1  # extra padding for fun :)
    # padding = len(right_perm) - 1
    left_padded = pad_permutation(left_perm, padding)
    full_size = len(left_padded)
    symmetric_group = sa.SymmetricGroup(full_size)
    sga = sa.GroupAlgebra(symmetric_group, sa.ZZ)

    accum = sga.zero()
    left_padded_sga = sga((left_padded + 1).tolist())
    for i in range(full_size - len(right_perm)):
        right_extended = np.arange(full_size, dtype=int)
        right_extended[i : i + len(right_perm)] = right_perm + i
        right_extended_sga = sga((right_extended + 1).tolist())
        comm = sga.bracket(left_padded_sga, right_extended_sga)
        accum += comm

    return accum, padding


def perm_compose_sided(left_perm, right_perm):
    assert is_valid_permutation(left_perm)
    assert is_valid_permutation(right_perm)
    sg = sa.SymmetricGroup(max(len(left_perm), len(right_perm)))
    left, right = sg((left_perm + 1).tolist()), sg((right_perm + 1).tolist())
    lr = list((left * right).tuple())
    rl = list((right * left).tuple())
    return lr, rl


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

    def selfsort_tuple(self):
        return (len(self.data), self.data.tolist())

    @staticmethod
    def bracket_homog_homog(left, right):
        left_perm, right_perm = left.data, right.data
        dragged, _ = drag_right_on_left(left_perm, right_perm)

        alg = left.alg or right.alg
        final = alg.zero()
        for sg_el, coeff in dragged:
            assert coeff != 0
            perm = list(sg_el.tuple())
            perm_promoted = alg(GLNHomogOp(perm))
            final += coeff * perm_promoted

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


class GLNBoostOp(GlobalOp):
    def __init__(self, perm, alg=None):
        perm = normalize_permutation(perm)
        assert is_reduced_permutation(perm)
        self.data = perm
        self.alg = alg

    def __repr__(self):
        return f"B[{repr_permutation(self.data)}]"

    def hardness(self):
        return 10

    def bracket_ordered(self, other):
        assert isinstance(other, GLNHomogOp)
        return GLNBoostOp.bracket_boost_homog(self, other)

    def selfsort_tuple(self):
        return (len(self.data), self.data.tolist())

    @staticmethod
    def boost(other):
        if isinstance(other, list) or isinstance(other, np.ndarray):
            return GLNBoostOp(other)
        if isinstance(other, GLNHomogOp):
            return GLNBoostOp(other.data, other.alg)
        elif other.parent() in sa.Modules:
            return extend_linear(GLNBoostOp.boost)(other)
        else:
            raise NotImplementedError()

    @staticmethod
    def reduce(perm, padding, alg):
        ring = alg.base()
        perm = normalize_permutation(perm)
        perm_squeezed = perm[padding : len(perm) - padding] - padding
        if is_reduced_permutation(perm_squeezed):
            return alg(GLNBoostOp(perm_squeezed))
        else:
            perm_reduced, legs = reduce_permutation(perm, padding=padding)
            left_legs, right_legs = legs
            left_legs, right_legs = ring(left_legs), ring(right_legs)
            accum = alg.zero()
            accum += alg(GLNBoostOp(perm_reduced))
            if symmetric_boost_identification:
                # the asymmetric identification is, IMO, much more intuitive,
                # but this is the one that matches eq. (3.20) in the paper
                left_legs, right_legs = (
                    (left_legs - right_legs) / 2,
                    (right_legs - left_legs) / 2,
                )
            accum -= left_legs * alg(GLNHomogOp(perm_reduced))
            accum -= right_legs * alg(GLNHomogOp([1]))
            return accum

    @staticmethod
    def bracket_boost_homog(left, right):
        # the methodology here is to write sum_a a sum_b [l(a), r(b)] and so
        # we're just expanding the inner sum and then calling that a boost op
        # (where we then need to do the appropriate operator identification)
        left_perm, right_perm = left.data, right.data
        dragged, padding = drag_right_on_left(left_perm, right_perm)

        alg = left.alg or right.alg
        final = alg.zero()
        for sg_el, coeff in dragged:
            final += coeff * GLNBoostOp.reduce(list(sg_el.tuple()), padding, alg)

        return final


class GLNBilocalOp(GlobalOp):
    r"""
    I really hate the paper's bilocal operator conventions, so I've rolled my
    own, and I hope it ends up working. I've included some scratch work below.

    Let's define $\antiwrap{A,B}\equiv\frac12\sum_a\Bqty{A_a,B_a}$ and
    $[A/B]\equiv\frac12\sum_a\sum_{b>a}(A_aB_b+B_bA_a)$. 
    $$
    \begin{align}
    [A/B]+[B/A]&=\frac12\sum_a\sum_{b>a}(A_aB_b+B_bA_a)+\frac12\sum_b\sum_{a>b}(A_aB_b+B_bA_a)\\
    &=\frac12\sum_a\sum_b\Bqty{A_a, B_b}-\frac12\sum_a\Bqty{A_a,B_a}\\
    [A|B]&=^?[A/B]+\frac12\antiwrap{A,B}\\
    [[A/B],C]&=\frac12\sum_a\sum_{b>a}\sum_c[\Bqty{A_a, B_b}, C_c]\\
    &=\frac12\sum_a\sum_{b>a}\sum_c(\Bqty{A_a, [B_b, C_c]}+\Bqty{B_b, [A_a, C_c]})\\
    $$
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
        if not isinstance(other, GLNHomogOp):
            raise NotImplementedError()
        return self.bracket_bilocal_homog(self, other)

    def selfsort_tuple(self):
        return (
            len(self.data[0]),
            len(self.data[1]),
            self.data[0].tolist(),
            self.data[1].tolist(),
        )

    @staticmethod
    def bilocalize(left, right):
        if isinstance(left, list) or isinstance(left, np.ndarray):
            assert isinstance(right, list) or isinstance(right, np.ndarray)
            return GLNBilocalOp(left, right)
        elif isinstance(left, GLNHomogOp):
            assert isinstance(right, GLNHomogOp)
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
            accum = alg.zero()
            for l, lc in left:
                for r, rc in right:
                    if not isinstance(l, GLNHomogOp) or not isinstance(r, GLNHomogOp):
                        breakpoint()
                    assert isinstance(l, GLNHomogOp) and isinstance(r, GLNHomogOp)
                    lx = np.arange(left_len, dtype=int)
                    try:
                        lx[: len(l.data)] = l.data
                    except:
                        breakpoint()
                    rx = np.arange(right_len, dtype=int)
                    rx[: len(r.data)] = r.data
                    accum += lc * rc * GLNBilocalOp.reduce_slashed(lx, 0, rx, 0, alg)[0]
            return accum

        else:
            raise NotImplementedError("Bilocalization for " + type(left))

    @staticmethod
    def reduce_slashed(left, lpad, rght, rpad, alg):
        r"""
        [A/B]&=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|} \Bqty{A_a, B_b}
        \\&=\frac12\sum_{0<a}\sum_{a<b\leq N-|B\leg |+1} \Bqty{A_a,  B \leg_b}
        \\&=\frac12\sum_{0<a}\sum_{a<b\leq N-|B\leg |} \Bqty{A_a, B\leg_b}+\frac12\sum_{0<a} \Bqty{A_a, B_{N-|B|}}
        \\&=[A/ B\leg ]+[A]\ev B
        \\&=\frac12\sum_{0<a}\sum_{a-1<b\leq N-|B|} \Bqty{A_a, \leg B_b}
        \\&=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|} \Bqty{A_a, \leg B_b}+\frac12\sum_a\Bqty{A_a, \leg B_a}
        \\&=[A/\leg B]+\antiwrap{A, \leg B}
        \\&=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|} \Bqty{A\leg_a,  B_b}
        \\&=[A\leg/ B]
        \\&=\frac12\sum_{-1<a}\sum_{a+1<b\leq N-|B|} \Bqty{\leg A_a, B_b}
        \\&=\frac12\sum_{0<a}\sum_{a<b\leq N-|B|} \Bqty{\leg A_a, B_b}+\frac12\sum_{b \leq N-|B|} \Bqty{A_0, B_b}-\frac12\sum_{0<a}\Bqty{\leg A_a, B_{a+1}}
        \\&=[\leg A/ B ]+\ev A [B]-\antiwrap{A, B}
        """
        left = normalize_permutation(left)
        rght = normalize_permutation(rght)
        left_red, left_legs = reduce_permutation(left, padding=lpad)
        ll_legs, lr_legs = left_legs

        rght_red, rght_legs = reduce_permutation(rght, padding=rpad)
        rl_legs, rr_legs = rght_legs

        left_hom, rght_hom = GLNHomogOp(left_red), GLNHomogOp(rght_red)
        primary = GLNBilocalOp(left_red, rght_red)

        accum = alg.zero()
        # accum = AccumWrapper(accum)
        accum += alg(primary)
        accum -= ll_legs * left_hom.vacuum_ev() * alg(rght_hom)
        accum -= rr_legs * rght_hom.vacuum_ev() * alg(left_hom)
        accum = GLNBilocalOp.reduce_slashed_recurse(
            left_red, ll_legs, rght_red, rl_legs, accum
        )
        # note that lr isn't used because I think it's only on boundary.

        accum = accum._wrapped if isinstance(accum, AccumWrapper) else accum
        return accum, primary

    @staticmethod
    def reduce_slashed_recurse(l_red, ll_legs, r_red, rl_legs, accum):
        alg = accum.parent()
        if rl_legs > 0:
            comms = perm_compose_sided(l_red, pad_permutation(r_red, rl_legs))
            accum -= GLNBilocalOp.antiwrap(comms, alg)
            return GLNBilocalOp.reduce_slashed_recurse(
                l_red, ll_legs, r_red, rl_legs - 1, accum
            )
        elif rl_legs < 0:
            comms = perm_compose_sided(l_red, r_red)
            accum += GLNBilocalOp.antiwrap(comms, alg)
            return GLNBilocalOp.reduce_slashed_recurse(
                l_red, ll_legs, pad_permutation(r_red, 1), rl_legs + 1, accum
            )
        elif ll_legs > 0:
            comms = perm_compose_sided(l_red, r_red)
            accum += GLNBilocalOp.antiwrap(comms, alg)
            return GLNBilocalOp.reduce_slashed_recurse(
                pad_permutation(l_red, 1), ll_legs - 1, r_red, rl_legs, accum
            )
        elif ll_legs < 0:
            comms = perm_compose_sided(l_red, pad_permutation(r_red, -ll_legs))
            accum -= GLNBilocalOp.antiwrap(comms, alg)
            return GLNBilocalOp.reduce_slashed_recurse(
                l_red, ll_legs + 1, r_red, rl_legs, accum
            )
        else:
            return accum

    @staticmethod
    def bracket_bilocal_homog(left, right):
        # it's possible that all the antiwrap terms will in general cancel out
        # in this method, but I haven't proved that it's true and it seems easy
        # to make a mistake while doing so.
        bileft, birght = left.data
        target = right.data
        # for short, these are bl, br, tg.

        alg = left.alg or right.alg
        accum_slashed = alg.zero()
        accum_antiwrap = alg.zero()
        br_tg_drag, br_tg_pad = drag_right_on_left(birght, target)
        for sg_el, coeff in br_tg_drag:
            br_tg_perm = list(sg_el.tuple())
            reduced, primary = GLNBilocalOp.reduce_slashed(
                bileft, 0, br_tg_perm, br_tg_pad, alg
            )
            accum_slashed += coeff * reduced

            red_left, red_rght = primary.data
            red_comms = perm_compose_sided(red_left, red_rght)
            accum_antiwrap -= coeff * GLNBilocalOp.antiwrap(red_comms, alg) / 2

        bl_tg_drag, bl_tg_pad = drag_right_on_left(bileft, target)
        for sg_el, coeff in bl_tg_drag:
            bl_tg_perm = list(sg_el.tuple())
            reduced, primary = GLNBilocalOp.reduce_slashed(
                bl_tg_perm, bl_tg_pad, birght, 0, alg
            )
            accum_slashed += coeff * reduced

            red_left, red_rght = primary.data
            red_comms = perm_compose_sided(red_left, red_rght)
            accum_antiwrap -= coeff * GLNBilocalOp.antiwrap(red_comms, alg) / 2

        bl_br_comms = perm_compose_sided(bileft, birght)
        accum_antiwrap += (
            GLNBilocalOp.antiwrap(bl_br_comms, alg).bracket(alg(right)) / 2
        )

        if not incl_antiwrap_in_bilocal_bracket:
            accum_antiwrap = 0
        return accum_slashed + accum_antiwrap

    @staticmethod
    def antiwrap(comms, alg):
        lr = alg(GLNHomogOp(comms[0]))
        rl = alg(GLNHomogOp(comms[1]))
        return (lr + rl) / 2
