def extend_linear(fn):
    return (
        lambda arg: fn(arg)
        if not hasattr(arg, "__iter__")
        else sum(coeff * arg.parent()(fn(elem)) for elem, coeff in arg)
    )


def extend_bilinear(fn):
    xtl = extend_linear
    return lambda left, right: xtl(lambda l: xtl(lambda r: fn(l, r))(right))(left)


def compose(*f):
    return (lambda *x: compose(*f[:-1])(f[-1](*x))) if f else lambda x: x
