from abc import ABC, abstractmethod
import numpy as np
import sage.all
from sage.algebras.lie_algebras.lie_algebra import InfinitelyGeneratedLieAlgebra
from sage.algebras.group_algebra import GroupAlgebra
from sage.categories.lie_algebras import LieAlgebras
from sage.combinat.all import CombinatorialFreeModule
from sage.rings.polynomial.all import PolynomialRing
from sage.groups.perm_gps.all import SymmetricGroup


def extend_linear(fn):
    # return (
    #     lambda arg: fn(arg)
    #     if not hasattr(arg, "__iter__")
    #     else sum(coeff * arg.parent().monomial(fn(elem)) for elem, coeff in arg)
    # )
    def inner(left):
        if not hasattr(left, "__iter__"):
            return fn(left)
        else:
            accum = left.parent().zero()
            for elem, coeff in left:
                fn_elem = fn(elem)
                accum += coeff * left.parent()(fn_elem)
            return accum

    return inner


def extend_bilinear(fn):
    xtl = extend_linear
    return lambda left, right: xtl(lambda l: xtl(lambda r: fn(l, r))(right))(left)


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
        if not isinstance(other, GlobalOp):
            breakpoint()
        assert isinstance(other, GlobalOp)
        if other.hardness() > self.hardness():
            return {k: -v for k, v in other.bracket_ordered(self).items()}
        else:
            return self.bracket_ordered(other)


class GlobalLengthOp(GlobalOp):
    def __init__(self):
        self.data = None

    def __repr__(self):
        return "N"

    def hardness(self):
        return 15

    def bracket_ordered(self, other):
        raise NotImplementedError()


class GlobalPermutationOp(GlobalOp):
    def __init__(self, perm):
        self.data = GlobalPermutationOp.reduce_permutation(perm)

    def __repr__(self):
        return f"{(self.data + 1).tolist()}".replace(" ", "")

    def hardness(self):
        return 1

    def bracket_ordered(self, other):
        assert isinstance(other, GlobalPermutationOp)
        return GlobalPermutationOp.bracket_perm_perm(self, other)

    @staticmethod
    def check_valid_permutation(perm):
        assert isinstance(perm, list) and all(isinstance(x, int) for x in perm)
        perm = np.array(perm, dtype=int)
        assert np.all(perm >= 1)
        assert np.all(perm <= len(perm))
        assert len(perm) == len(set(perm))

    @staticmethod
    def reduce_permutation(perm):
        GlobalPermutationOp.check_valid_permutation(perm)
        perm = np.array(perm, dtype=int)
        perm -= 1  # Convert to zero-based indexing
        nontrivial_indices = np.where(perm != np.arange(len(perm)))[0]
        if len(nontrivial_indices) == 0:
            return np.array([0], dtype=int)
        perm = perm[nontrivial_indices.min() : nontrivial_indices.max() + 1]
        perm -= perm.min()
        GlobalPermutationOp.check_valid_permutation((perm + 1).tolist())
        return perm

    @staticmethod
    def pad_permutation(perm, padding):
        larger_perm = np.arange(len(perm) + 2 * padding, dtype=int)
        larger_perm[padding:-padding] = perm + padding
        return larger_perm

    @staticmethod
    def bracket_perm_perm(left, right):
        left_perm, right_perm = left.data, right.data
        padding = len(left_perm) + 1
        right_padded = GlobalPermutationOp.pad_permutation(right_perm, padding)
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
            perm_promoted = GlobalPermutationOp(perm)
            pre_coeff = final_dict.get(perm_promoted, 0)
            final_dict[perm_promoted] = pre_coeff + coeff

        return final_dict


class GlobalBoostOp(GlobalOp):
    def __init__(self, child):
        if not isinstance(child, GlobalPermutationOp):
            breakpoint()
        assert isinstance(child, GlobalPermutationOp)
        self.data = child

    def __repr__(self):
        return f"B[{self.data}]"

    def hardness(self):
        return 10

    def bracket_ordered(self, other):
        assert isinstance(other, GlobalPermutationOp)
        return GlobalBoostOp.bracket_boost_perm(self, other)

    @staticmethod
    def boost(other):
        if isinstance(other, GlobalPermutationOp):
            return GlobalBoostOp(other)
        elif isinstance(other.parent(), GlobalAlgebra):
            return extend_linear(GlobalBoostOp)(other)
        else:
            raise NotImplementedError()

    @staticmethod
    def bracket_boost_perm(left, right):
        Kpoly = PolynomialRing(sage.all.ZZ, "k")
        localization_offset = Kpoly.gen()
        left_perm, right_perm = left.data.data, right.data
        padding = len(left_perm) + 1
        right_padded = GlobalPermutationOp.pad_permutation(right_perm, padding)
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
            # this max() bit is really subtle: it has to do with the fact that
            # if the left permutation starts before the right (which we've
            # localized), we aren't actually acting at the site of that
            # localized right permutation, so we need to think about which
            # site we're actually acting on and use the appropriate shift.
            # also, since the localization offset is arbitrary, I think we
            # could have done max(i, padding) instead, but we're doing it this
            # way to more directly reflect the choice of spin chain origin.
            accum += (localization_offset + max(i - padding, 0)) * (lr - rl)

        final_dict = {}
        for sg_el, coeff in accum:
            assert coeff != 0
            perm = list(sg_el.tuple())
            perm_promoted = GlobalPermutationOp(perm)
            pre_coeff = final_dict.get(perm_promoted, 0)
            final_dict[perm_promoted] = pre_coeff + coeff

        return final_dict


class GlobalBilocalOp(GlobalOp):
    def __init__(self, left, right):
        self.data = (left, right)

    def __repr__(self):
        left, right = self.data
        return f"[ {left} | {right} ]"

    def hardness(self):
        return 100

    def bracket_ordered(self, other):
        raise NotImplementedError()


def operators_ordered(first, second):
    op_order = [GlobalBilocalOp, GlobalBoostOp, GlobalLengthOp, GlobalPermutationOp]
    return op_order.index(type(first)) <= op_order.index(type(second))


class GlobalAlgebra(CombinatorialFreeModule):
    def __init__(self):
        raw_ring = sage.all.QQbar
        self.i_ = raw_ring.gen()
        ring = PolynomialRing(raw_ring, "k")
        category = LieAlgebras(ring)
        super().__init__(ring, basis_keys=None, category=category, prefix="e")

    def _repr_(self):
        return "Global#Algebra"

    def _element_constructor_(self, x):
        """Convert x into an element of this algebra"""
        if isinstance(x, GlobalOp):
            return self.monomial(x)
        elif isinstance(x, dict):
            result = self.zero()
            for op, coeff in x.items():
                result += coeff * self(op)
            return result
        else:
            return super()._element_constructor_(x)

    def bracket(self, left_op, right_op):
        """Compute the bracket of two basis elements"""
        result = self(extend_bilinear(GlobalOp._bracket_)(left_op, right_op))
        for op, coeff in result:
            if coeff.degree() > 0:
                breakpoint()
            assert coeff.degree() == 0
        return result


def do_the_thing():
    global_algebra = GlobalAlgebra()
    i_ = global_algebra.i_

    test = GlobalPermutationOp([1, 3, 2])
    print("Test permutation:", test)

    def GPO(perm):
        return global_algebra(GlobalPermutationOp(perm))

    Q2 = GPO([1]) - GPO([2, 1])
    BQ2 = GlobalBoostOp.boost(Q2)

    print("Q2:", Q2)
    print("BQ2:", BQ2)
    print("[Q2, Q2]:", global_algebra.bracket(Q2, Q2))
    print("[BQ2, Q2]:", global_algebra.bracket(BQ2, Q2))

    charge_top = 5
    charge_stack = [None, None, Q2]
    for k in range(2, charge_top):
        charge_stack.append(global_algebra.bracket(BQ2, charge_stack[-1]) * -i_ / k)
        print(f"Q{k + 1}:", charge_stack[-1])

    breakpoint()


if __name__ == "__main__":
    print("starting main")
    do_the_thing()
    print("finished main")
