from abc import ABC, abstractmethod


class SpinChain(ABC):
    @abstractmethod
    def Q(self, k):
        pass

    @abstractmethod
    def bracket(self, left, right):
        pass

    @abstractmethod
    def algebra(self):
        pass

    @abstractmethod
    def format(self, q):
        pass

    @abstractmethod
    def top_charge(self):
        pass

    def homogeneity(self, q=None):
        if q is None:
            for q in range(2, len(self.charge_tower)):
                if not self.homogeneity(q):
                    return False
            return True
        cur_charge = self.Q(q)
        consist_type = type([el for el, _ in self.algebra().make([1])][0])
        for el, _ in cur_charge:
            if type(el) is not consist_type:
                return False
        return True

    def algebra_consistency(self):
        for q in range(2, self.top_charge() + 1):
            for r in range(2, q):
                bracket = self.bracket(self.Q(q), self.Q(r))
                if not bracket.is_zero():
                    bracket_str = str(bracket)
                    if len(bracket_str) > 50:
                        bracket_str = bracket_str[:50] + "..."
                    print(
                        f"Algebra inconsistency at order {self.order()}: "
                        f"[Q{q}, Q{r}] = {bracket_str} != 0"
                    )
                    return False
        return True


class BaseChain(SpinChain):
    def __init__(self, hamiltonian, logging=False):
        self.hamiltonian = hamiltonian
        self.alg = hamiltonian.parent()
        alg = self.alg
        self.i_ = alg.i_
        length_op = alg.make([1])  # calling this Q1 isn't standard, but it feels right
        self.charge_tower = [None, length_op, hamiltonian]
        self.boost_tower = [None, alg.boost(length_op), alg.boost(hamiltonian)]
        self.logging = logging

    def Q(self, k):
        assert k >= 1, "Charges are defined for k >= 1"
        self.ensure_filled(k)
        return self.charge_tower[k]

    def BQ(self, k):
        assert k >= 1, "Boosts are defined for k >= 1"
        self.ensure_filled(k)
        return self.boost_tower[k]

    def ensure_filled(self, k):
        if k < len(self.charge_tower):
            return
        self.ensure_filled(k - 1)
        BQ2_bracket_Qtop = self.boost_tower[2].bracket(self.charge_tower[k - 1])
        self.charge_tower.append(BQ2_bracket_Qtop * -self.i_ / (k - 1))
        self.boost_tower.append(self.alg.boost(self.charge_tower[k]))
        if self.logging:
            print(f"Q_{k}: 1/{k - 1}*({(k - 1) * self.charge_tower[k]})")

    def bracket(self, left, right):
        return left.bracket(right)

    def algebra(self):
        return self.alg

    def format(self, q):
        return str(q)

    def top_charge(self):
        return len(self.charge_tower) - 1

    def first_order_boost_deformation_reduced(self, r, k):
        raise DeprecationWarning()
        # implements eq (B.4)
        BQ, Q = self.BQ, self.Q
        return (k - 1) * self.i_ * BQ(k).bracket(Q(r)) + (r + k - 2) * Q(r + k - 1)
        # can we instead do the above by constructing a left side that just
        # hits Qr? (we'd probably need to implement the G rotation generator)
