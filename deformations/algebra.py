from abc import ABC, abstractmethod
from functools import total_ordering
from operator import is_

import numpy as np
import sage.all as sa
from sage.algebras.lie_algebras.lie_algebra import LieAlgebraWithGenerators
from sage.algebras.lie_algebras.lie_algebra_element import LieAlgebraElement
from sage.parallel.decorate import parallel
from sage.structure.indexed_generators import IndexedGenerators

from .config import enable_logging, log_basis_bracket
from .tools import compose, is_zero, map_collect_elements, wrap_logging

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
        if type(result) is dict:
            neg_result = {k: -v for k, v in result.items()}
        else:
            neg_result = -result
        if cache:
            GlobalOp._bracket_cache[cache_key] = (result, neg_result)
        return (result, neg_result)

    @staticmethod
    def update_bracket_cache(kv_list):
        for cache_key, result in kv_list:
            if cache_key not in GlobalOp._bracket_cache:
                GlobalOp._bracket_cache[cache_key] = result

    @profile
    def _bracket_(self, other, cache=True):
        assert isinstance(other, GlobalOp)
        if other.sort_order() < self.sort_order():
            bkt = other.bracket_ordered_cached(self, cache=cache)
            return bkt[1]
        else:
            bkt = self.bracket_ordered_cached(other, cache=cache)
            return bkt[0]

    def bracket(self, other, cache=True):
        return self._bracket_(other, cache=cache)

    def sort_order(self):
        return (-self.hardness(), *self.selfsort_tuple())

    @abstractmethod
    def selfsort_tuple(self):
        pass

    def set_alg(self, alg):
        self.alg = alg
        return self


class GlobalAlgebra(sa.IndexedGenerators, sa.LieAlgebraWithGenerators):
    def __init__(self, bilocalize, boost=None, make=None, ring=None, i_=None):
        # raw_ring = sa.QQbar
        # self.i_ = self.raw_ring.gen()
        if ring is None:
            raw_ring = (sa.QQ(1) * sa.I).parent()
            self.i_ = raw_ring.gen()
            ring = raw_ring
        else:
            assert i_ is not None
            self.i_ = i_
        self.ring = ring
        cat = sa.LieAlgebras(ring).WithBasis()

        self.boost = compose(self, boost) if boost is not None else self.bilocal_boost
        self.bilocalize = compose(self, bilocalize)
        self.make = compose(self, make)

        self._repr_term = str
        self.bracket_on_basis = compose(self, GlobalOp._bracket_)
        if enable_logging and log_basis_bracket:
            self.bracket_on_basis = wrap_logging(self.bracket_on_basis)
        sa.Parent.__init__(self, base=ring, category=cat)
        sa.IndexedGenerators.__init__(self, indices=(), prefix="")
        self.print_options(prefix="", bracket="")

    def _repr_(self):
        return "GlobalAlgebra over " + str(self.ring)

    @profile
    def _element_constructor_(self, x):
        """Convert x into an element of this algebra"""
        if type(x) is dict:
            for k, v in tuple(x.items()):
                if is_zero(k) or is_zero(v):
                    del x[k]
        elif isinstance(x, GlobalOp):
            x.alg = self
            x = {x: self.ring(1)}
        elif is_zero(x):
            return self.zero()
        elif hasattr(x, "parent") and isinstance(x.parent(), GlobalAlgebra):
            return self._element_constructor_({k.set_alg(self): v for k, v in x})
        else:
            raise TypeError(f"Cannot convert {type(x)} to an element of GlobalAlgebra.")
        return sa.LieAlgebraWithGenerators._element_constructor_(self, x)

    def bilocal_boost(self, Q):
        ones = self.make([1])
        return (self.bilocalize(ones, Q) - self.bilocalize(Q, ones)) / 2

    def with_variable(self, name="λ"):
        raw_ring = self.ring
        if name in [str(g) for g in raw_ring.gens()]:
            old_name, name = name, name + str(len(raw_ring.gens()))
            print(
                f"{old_name} already in generator list {raw_ring.gens()}, using {name} instead."
            )
        nested_ring = sa.PolynomialRing(raw_ring, 1, name)
        var = nested_ring.gen()
        ring = nested_ring.flattening_morphism().codomain()
        alg = GlobalAlgebra(
            self.bilocalize, boost=self.boost, make=self.make, ring=ring, i_=self.i_
        )
        return alg, var

    class Element(sa.LieAlgebraElement):
        @profile
        def _bracket_(self, other):
            alg = self.parent()
            brackets_list = [
                (k1, k2, (v1 * v2)) for (k1, v1) in self for (k2, v2) in other
            ]
            brackets_list = [
                (GlobalOp.bracket(k1, k2), v) for k1, k2, v in brackets_list
            ]
            brackets_list = [
                r for r in brackets_list if not is_zero(r[0]) and not is_zero(r[1])
            ]
            brackets_list = [
                (k, v * v2) for b, v2 in brackets_list for k, v in b.items()
            ]
            brackets = map_collect_elements(
                brackets_list, lambda k, v: (k, v), alg.ring.zero()
            )
            return alg(brackets)

        pass
