from .spin_chains import SpinChain
from .tools import extend_to_coeffs, map_collect_elements

try:
    from line_profiler import profile
except ImportError:
    profile = lambda *_, **__: lambda fn: fn


class DeformedChain(SpinChain):
    def __init__(self, base_chain, deformation, deform_param="λ"):
        self.base_chain = base_chain
        self.alg, self.param = base_chain.algebra().with_variable(deform_param)
        self.i_ = self.alg.i_
        self.charge_tower = [None] + [self.alg(q) for q in base_chain.charge_tower[1:]]
        self.orders = [None] + [0] * (len(self.charge_tower) - 1)
        self.deform_lam = self.deformation_lambda(deformation)
        self.ensure_filled(max(deformation))

    def Q(self, k):
        assert k >= 1, "Charges are defined for k >= 1"
        self.ensure_filled(k)
        return self.charge_tower[k]

    def ensure_filled(self, q):
        if q < len(self.charge_tower):
            return
        self.ensure_filled(q - 1)
        self.orders.append(0)
        self.charge_tower.append(self.alg(self.base_chain.Q(q)))
        self.ensure_order(self.orders[1], q=q)

    def order(self):
        return self.orders[1]

    def ensure_order(self, k, q=None):
        if q is None:
            if k <= self.orders[1]:
                return
            self.ensure_order(k - 1)
            for q in range(2, len(self.charge_tower)):
                self.ensure_order(k, q)
            self.orders[1] = k
            return
        if k <= self.orders[q]:
            return
        self.ensure_order(k - 1, q)
        cur_charge = self.charge_tower[q]
        gen = self.deform_gen()
        integrand = self.i_ * self.bracket_at_order(
            gen, cur_charge, k - 1, promote=True
        )
        integral = integrand * ((self.param**k) / k)
        self.charge_tower[q] = cur_charge + integral
        if not self.homogeneity(q):
            raise ValueError(
                f"Homogeneity check failed for charge {q} at order {k}. "
                "This may indicate an inconsistency in the algebra. "
                f"Q_{q} string starts with {str(self.charge_tower[q])[:200]}..."
            )
        print(f"Charge {q} at order {k} has {len(self.charge_tower[q])} terms.")
        self.orders[q] = k

    def deformation_lambda(self, deformation):
        match deformation:
            case f if callable(f):
                return f
            case (k,) if type(k) is int:
                self.ensure_filled(k)
                return lambda self: self.alg.boost(self.Q(k))
            case (k, -1) if type(k) is int:
                self.ensure_filled(k)
                return lambda self: self.alg.bilocal_boost(self.Q(k))
            case (k, l) if type(k) is int and type(l) is int:
                self.ensure_filled(k)
                self.ensure_filled(l)
                return lambda self: self.alg.bilocalize(self.Q(k), self.Q(l))
            case _:
                raise NotImplementedError()

    def deform_gen(self):
        return self.deform_lam(self)

    def bracket_to_order(self, left, right, order=None):
        left_ords = self.extract_orders(left)
        right_ords = self.extract_orders(right)
        if order is None:
            order = self.order()
        res = 0
        for k in range(order + 1):
            res += self.param**k * self.bracket_at_order(
                left_ords, right_ords, k, promote=True
            )
        return res

    @profile
    def bracket_at_order(self, left, right, order, promote=True):
        if type(left) is not list:
            left = self.extract_orders(left)
        if type(right) is not list:
            right = self.extract_orders(right)
        res = []
        for k in range(order + 1):
            if k < len(left) and order - k < len(right):
                res.append(self.base_chain.bracket(left[k], right[order - k]))
        if promote:
            return self.alg.sum(res)
        else:
            return self.base_chain.algebra().sum(res)

    def extract_orders(self, q):
        if q == 0:
            return [q]

        terms_lists = [[]]
        ring = self.alg.ring
        lower_ring = self.base_chain.algebra().ring
        for el, coeff in q:
            coeff_deg = ring(coeff).degrees()[-1]
            if coeff_deg == 0:
                terms_lists[0].append((el, coeff))
                continue
            other_gens = lower_ring.gens()
            while coeff_deg >= len(terms_lists):
                terms_lists += [[]]
            processed = []
            for k, v in coeff.dict().items():
                neg_one = len(k) - 1  # sage can't [-1]
                degree = k[neg_one]
                other_degrees = k[:neg_one]
                acc = lower_ring(v)
                for d, g in zip(other_degrees, other_gens):
                    acc *= g**d
                processed.append((degree, acc))
            for d, coeff_d in processed:
                terms_lists[d].append((el, coeff_d))

        zero = lower_ring.zero()
        terms = []
        for d in range(len(terms_lists)):
            mapped = map_collect_elements(terms_lists[d], lambda k, v: (k, v), zero)
            terms.append(self.base_chain.algebra()(mapped))

        return terms

    def truncate_order(self, q, order=None):
        if order is None:
            order = self.order()
        return extend_to_coeffs(lambda x: x % (self.param ** (order + 1)))(q)

    def format(self, q, order=None):
        if order is None:
            order = self.order()
        terms = self.extract_orders(q)
        res = str(terms[0])
        for k in range(1, len(terms)):
            if terms[k] != 0:
                res += f" + {self.param}^{k}*({self.base_chain.format(terms[k])})"
        max_order = max(len(terms) - 1, order or 0)
        res += f" + O({self.param}^{max_order + 1})"
        return res

    def bracket(self, left, right):
        return self.bracket_to_order(left, right, self.order())

    def algebra(self):
        return self.alg

    def top_charge(self):
        return len(self.charge_tower) - 1
