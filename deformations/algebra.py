from abc import ABC, abstractmethod
import numpy as np
import sage.all as sa

from .tools import extend_bilinear


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


class GlobalAlgebra(sa.CombinatorialFreeModule):
    def __init__(self, boost, make=None):
        self.boost_ = boost
        self.boost = lambda x: self(boost(x))
        self.make = lambda x: self(make(x))  # for convenience
        raw_ring = sa.QQbar
        self.i_ = raw_ring.gen()
        ring = sa.PolynomialRing(raw_ring, "k")
        category = sa.LieAlgebras(ring)
        super().__init__(ring, basis_keys=None, category=category)
        self.print_options(sorting_key=lambda x: x.sort_order(), prefix="", bracket="")

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

    # this is a hack to make a.bracket(b) work
    class Element(sa.CombinatorialFreeModule.Element):
        def _bracket_(self, right):
            return self.parent().bracket(self, right)
