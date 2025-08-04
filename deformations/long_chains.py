from deformations.tools import extend_to_coeffs


class LongRangeChain:
    def __init__(self, short_range_chain, deformation):
        self.base_chain = short_range_chain
        self.alg = short_range_chain.alg
        self.i_ = self.alg.i_
        self.deform_param = self.alg.ring.gen()
        self.charge_tower = short_range_chain.charge_tower.copy()
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
        self.charge_tower.append(self.base_chain.Q(q))
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
        self.charge_tower[q] = cur_charge + extend_to_coeffs(
            lambda x: (x * self.deform_param ** (k - 1)).integral()
        )(self.i_ * self.bracket_at_order(gen, cur_charge, k - 1))
        assert self.homogeneity(q)
        self.orders[q] = k

    def deformation_lambda(self, deformation):
        if callable(deformation):
            return deformation
        assert isinstance(deformation, tuple)
        assert isinstance(deformation[0], int)
        if len(deformation) == 1:
            (k,) = deformation
            self.ensure_filled(k)
            return lambda self: self.alg.boost(self.Q(k))
        elif len(deformation) == 2:
            k, l = deformation
            self.ensure_filled(k)
            self.ensure_filled(l)
            return lambda self: self.alg.bilocalize(self.Q(k), self.Q(l))
        else:
            raise NotImplementedError()

    def deform_gen(self):
        return self.deform_lam(self)

    @staticmethod
    def bracket_to_order(left, right, order=None):
        left_ords = LongRangeChain.extract_orders(left)
        right_ords = LongRangeChain.extract_orders(right)
        if order is None:
            order = max(len(left_ords) - 1, len(right_ords) - 1)
        res = 0
        lam = left.parent().base_ring().gen()
        assert lam != 0
        for k in range(order + 1):
            res += lam**k * LongRangeChain.bracket_at_order(left_ords, right_ords, k)
        return res

    @staticmethod
    def bracket_at_order(left, right, order):
        if not isinstance(left, list):
            left = LongRangeChain.extract_orders(left)
        if not isinstance(right, list):
            right = LongRangeChain.extract_orders(right)
        res = 0
        for k in range(order + 1):
            if k < len(left) and order - k < len(right):
                res += left[k].bracket(right[order - k])
        return res

    def homogeneity(self, q=None):
        if q is None:
            for q in range(2, len(self.charge_tower)):
                if not self.homogeneity(q):
                    return False
            return True
        if q < 2:
            return True
        cur_charge = self.charge_tower[q]
        consist_type = type([el for el, _ in self.alg.make([1])][0])
        for el, _ in cur_charge:
            if not isinstance(el, consist_type):
                return False
        return True

    def algebra_consistency(self):
        for q in range(2, len(self.charge_tower)):
            for r in range(2, q):
                bracket = self.bracket_to_order(self.Q(q), self.Q(r), self.order())
                if bracket != 0:
                    return False
        return True

    @staticmethod
    def extract_orders(q):
        if q == 0:
            return [0]
        terms = [0]
        for el, coeff in q:
            terms += [0] * (coeff.degree() + 1 - len(terms))
            for d in range(len(terms)):
                terms[d] += coeff.coefficient(d) * q.parent()(el)
        return terms

    def truncate_order(self, q, order=None):
        if order is None:
            order = self.order()
        lam = q.parent().base_ring().gen()
        return extend_to_coeffs(lambda x: x % (lam ** (order + 1)))(q)

    @staticmethod
    def format(q, order=None):
        if order is not None:
            q = LongRangeChain.truncate_order(None, q, order)
        terms = LongRangeChain.extract_orders(q)
        res = str(terms[0])
        for k in range(1, len(terms)):
            if terms[k] != 0:
                res += f" + λ^{k}*({terms[k]})"
        max_order = max(len(terms) - 1, order or 0)
        res += f" + O(λ^{max_order + 1})"
        return res
