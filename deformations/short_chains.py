import sage.all as sa


class ShortRangeChain:
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

    def first_order_boost_deformation_reduced(self, r, k):
        # implements eq (B.4)
        BQ, Q = self.BQ, self.Q
        return (k - 1) * self.i_ * BQ(k).bracket(Q(r)) + (r + k - 2) * Q(r + k - 1)
        # can we instead do the above by constructing a left side that just
        # hits Qr? (we'd probably need to implement the G rotation generator)
