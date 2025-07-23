import sage.all
from sage.categories.lie_algebras import LieAlgebras
from sage.combinat.all import CombinatorialFreeModule
from sage.rings.polynomial.all import PolynomialRing

from .operators import (
    GlobalOp,
    extend_bilinear,
)


class GlobalGLNAlgebra(CombinatorialFreeModule):
    def __init__(self):
        raw_ring = sage.all.QQbar
        self.i_ = raw_ring.gen()
        ring = PolynomialRing(raw_ring, "k")
        category = LieAlgebras(ring)
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
    class Element(CombinatorialFreeModule.Element):
        def _bracket_(self, right):
            return self.parent().bracket(self, right)
