import sage.all as sa


class ShortRangeChain:
    def __init__(self, hamiltonian, logging=False):
        self.hamiltonian = hamiltonian
        self.algebra = hamiltonian.parent()
        self.i_ = self.algebra.i_
        self.charge_tower = [None, None, hamiltonian]
        self.boost_tower = [None, None, self.algebra.boost(hamiltonian)]
        self.logging = logging

    def Q(self, k):
        if k < 2:
            raise IndexError("k must be at least 2")
        self.ensure_filled(k)
        return self.charge_tower[k]

    def BQ(self, k):
        if k < 2:
            raise IndexError("k must be at least 2")
        self.ensure_filled(k)
        return self.boost_tower[k]

    def ensure_filled(self, k):
        if k < len(self.charge_tower):
            return
        self.ensure_filled(k - 1)
        BQ2_bracket_Qtop = self.boost_tower[2].bracket(self.charge_tower[k - 1])
        self.charge_tower.append(BQ2_bracket_Qtop * -self.i_ / (k - 1))
        self.boost_tower.append(self.algebra.boost(self.charge_tower[k]))
        if self.logging:
            print(f"Q_{k}: 1/{k - 1}*({(k - 1) * self.charge_tower[k]})")

    def first_order_boost_deformation_reduced(self, r, k):
        # implements eq (B.4)
        return (k - 1) * self.i_ * self.BQ(k).bracket(self.Q(r)) + (r + k - 2) * self.Q(
            r + k - 1
        )
        # can we instead do the above by constructing a left side that just
        # hits Qr? (we'd probably need to implement the G rotation generator)
