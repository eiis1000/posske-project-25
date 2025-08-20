from abc import ABC, abstractmethod
from functools import total_ordering

import numpy as np
import sage.all as sa
from sage.algebras.lie_algebras.lie_algebra import LieAlgebraWithGenerators
from sage.algebras.lie_algebras.lie_algebra_element import LieAlgebraElement
from sage.structure.indexed_generators import IndexedGenerators

from .config import enable_logging, log_basis_bracket
from .tools import compose, wrap_logging

# from sage.parallel.multiprocessing_sage import parallel_iter
# from sage.parallel.map_reduce import RESetMapReduce

sa.IndexedGenerators = IndexedGenerators
sa.LieAlgebraWithGenerators = LieAlgebraWithGenerators
sa.LieAlgebraElement = LieAlgebraElement
# sa.parallel_iter = parallel_iter
# sa.RESetMapReduce = RESetMapReduce

try:
    from line_profiler import profile
except ImportError:
    profile = lambda *_, **__: lambda fn: fn


_bracket_cache = {}


@total_ordering
class GlobalOp(ABC):
    _hash = None

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
    def bracket_ordered_cached(self, other):
        cache_key = (hash(self), hash(other))
        if cache_key in _bracket_cache:
            return _bracket_cache[cache_key]
        result = self.bracket_ordered(other)
        _bracket_cache[cache_key] = result
        return result

    def _bracket_(self, other):
        assert isinstance(other, GlobalOp)
        if other.sort_order() < self.sort_order():
            return {k: -v for k, v in other.bracket_ordered_cached(self)}
        else:
            return self.bracket_ordered_cached(other)

    def bracket(self, other):
        return self._bracket_(other)

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
        else:
            raise TypeError(f"Cannot convert {type(x)} to an element of GlobalAlgebra.")
        return sa.LieAlgebraWithGenerators._element_constructor_(self, x)

    def bilocal_boost(self, Q):
        ones = self.make([1])
        return (self.bilocalize(ones, Q) - self.bilocalize(Q, ones)) / 2

    class Element(sa.LieAlgebraElement):
        # def _bracket_(self, other):
        #     # left_items = {k: v for k, v in self}
        #     # right_items = {k: v for k, v in other}
        #     collected_items = [
        #         (k1, k2, v1 * v2) for (k1, v1) in self for (k2, v2) in other
        #     ]
        #     alg = self.parent()
        #     S = RESetMapReduce(
        #         roots=collected_items,
        #         children=lambda _: (),
        #         map_function=lambda x: x[2] * alg(GlobalOp._bracket_(x[0], x[1])),
        #         reduce_function=lambda x, y: x + y,
        #         reduce_init=alg.zero(),
        #     )
        #     return S.run()

        pass
