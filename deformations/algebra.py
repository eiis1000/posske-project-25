from abc import ABC, abstractmethod
from functools import total_ordering

import numpy as np
import sage.all as sa
from sage.algebras.lie_algebras.lie_algebra import LieAlgebraWithGenerators
from sage.algebras.lie_algebras.lie_algebra_element import LieAlgebraElement
from sage.parallel.decorate import parallel
from sage.structure.indexed_generators import IndexedGenerators

from .config import enable_logging, log_basis_bracket
from .tools import compose, wrap_logging

sa.LieAlgebraWithGenerators = LieAlgebraWithGenerators
sa.LieAlgebraElement = LieAlgebraElement
sa.parallel = parallel
sa.IndexedGenerators = IndexedGenerators

try:
    from line_profiler import profile
except ImportError:
    profile = lambda *_, **__: lambda fn: fn


@total_ordering
class GlobalOp(ABC):
    _hash = None  # will get shadowed by local variable
    _bracket_cache = {}  # actually global

    def __hash__(self):
        if self._hash is not None:
            return self._hash
        if type(self.data) is np.ndarray:
            self._hash = hash((self.data.tobytes(), type(self)))
        elif type(self.data) is list or type(self.data) is tuple:
            self._hash = hash((tuple(k.tobytes() for k in self.data), type(self)))
        else:
            self._hash = hash((self.data, type(self)))
        return self._hash

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        if type(self.data) is np.ndarray and type(other.data) is np.ndarray:
            return np.array_equal(self.data, other.data)
        elif type(self.data) is tuple:
            assert len(self.data) == 2
            assert type(self.data[0]) is np.ndarray
            self_l, self_r = self.data
            other_l, other_r = other.data
            return np.array_equal(self_l, other_l) and np.array_equal(self_r, other_r)
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

    @profile
    def bracket_ordered_cached(self, other, cache=True):
        cache_key = (hash(self), hash(other))
        if cache_key in GlobalOp._bracket_cache:
            return GlobalOp._bracket_cache[cache_key]
        result = self.bracket_ordered(other)
        if cache:
            GlobalOp._bracket_cache[cache_key] = result
        return result

    @staticmethod
    def update_bracket_cache(kv_list):
        for cache_key, result in kv_list:
            if cache_key not in GlobalOp._bracket_cache:
                GlobalOp._bracket_cache[cache_key] = result

    def _bracket_(self, other, cache=True):
        assert isinstance(other, GlobalOp)
        if other.sort_order() < self.sort_order():
            bkt = other.bracket_ordered_cached(self)
            if type(bkt) is dict:
                return {k: -v for k, v in bkt.items()}
            else:
                return -bkt
        else:
            return self.bracket_ordered_cached(other)

    def bracket(self, other, cache=True):
        return self._bracket_(other, cache=cache)

    def sort_order(self):
        return (-self.hardness(), *self.selfsort_tuple())

    @abstractmethod
    def selfsort_tuple(self):
        pass


class GlobalAlgebra(sa.IndexedGenerators, sa.LieAlgebraWithGenerators):
    def __init__(self, boost, bilocalize, make=None):
        self.boost_term = boost
        self.boost = compose(self, boost) if boost is not None else self.bilocal_boost
        self.bilocalize = compose(self, bilocalize)
        self.make = compose(self, make)
        # raw_ring = sa.QQbar
        # self.i_ = self.raw_ring.gen()
        raw_ring = (sa.QQ(1) * sa.I).parent()
        self.i_ = raw_ring.gen()
        ring = sa.PolynomialRing(raw_ring, "λ")
        self.ring = ring
        cat = sa.LieAlgebras(ring).WithBasis()

        self._repr_term = str
        self.bracket_on_basis = compose(self, GlobalOp._bracket_)
        if enable_logging and log_basis_bracket:
            self.bracket_on_basis = wrap_logging(self.bracket_on_basis)
        sa.Parent.__init__(self, base=ring, category=cat)
        sa.IndexedGenerators.__init__(self, indices=(), prefix="")
        self.print_options(prefix="", bracket="")

    def _repr_(self):
        return "Global#Algebra"

    @profile
    def _element_constructor_(self, x):
        """Convert x into an element of this algebra"""
        if type(x) is dict:
            for k, v in tuple(x.items()):
                if v == 0:
                    del x[k]
        elif isinstance(x, GlobalOp):
            x.alg = self
            x = {x: self.ring(1)}
        elif x == 0:
            return self.zero()
        else:
            raise TypeError(f"Cannot convert {type(x)} to an element of GlobalAlgebra.")
        return sa.LieAlgebraWithGenerators._element_constructor_(self, x)

    def bilocal_boost(self, Q):
        ones = self.make([1])
        return (self.bilocalize(ones, Q) - self.bilocalize(Q, ones)) / 2

    class Element(sa.LieAlgebraElement):
        pass
