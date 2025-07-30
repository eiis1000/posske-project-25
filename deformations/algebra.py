from abc import ABC, abstractmethod
from functools import total_ordering
import numpy as np
import sage.all as sa

from .tools import compose, wrap_logging

from sage.structure.indexed_generators import IndexedGenerators
from sage.algebras.lie_algebras.lie_algebra import LieAlgebraWithGenerators
from sage.algebras.lie_algebras.lie_algebra_element import LieAlgebraElement

sa.IndexedGenerators = IndexedGenerators
sa.LieAlgebraWithGenerators = LieAlgebraWithGenerators
sa.LieAlgebraElement = LieAlgebraElement


@total_ordering
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

    def __lt__(self, other):
        return self.sort_order() < other.sort_order()

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
        if other.sort_order() < self.sort_order():
            return {k: -v for k, v in other.bracket_ordered(self).items()}
        else:
            return self.bracket_ordered(other)

    def sort_order(self):
        return (-self.hardness(), *self.selfsort_tuple())

    @abstractmethod
    def selfsort_tuple(self):
        pass


class GlobalAlgebra(sa.IndexedGenerators, sa.LieAlgebraWithGenerators):
    def __init__(self, boost, make=None):
        self.boost_term = boost
        self.boost = compose(self, boost)
        self.make = compose(self, make)
        raw_ring = sa.QQbar
        self.i_ = raw_ring.gen()
        ring = sa.PolynomialRing(raw_ring, "k")
        cat = sa.LieAlgebras(ring).WithBasis()

        self._repr_term = str
        self.bracket_on_basis = compose(self, GlobalOp._bracket_)
        self.bracket_on_basis = wrap_logging(self.bracket_on_basis)
        sa.Parent.__init__(self, base=ring, category=cat)
        sa.IndexedGenerators.__init__(self, indices=(), prefix="")
        self.print_options(prefix="", bracket="")

    def _repr_(self):
        return "Global#Algebra"

    def _element_constructor_(self, x):
        """Convert x into an element of this algebra"""
        if isinstance(x, GlobalOp):
            x = {x: 1}
        return sa.LieAlgebraWithGenerators._element_constructor_(self, x)

    class Element(sa.LieAlgebraElement):
        pass
