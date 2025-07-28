def extend_linear(fn):
    return (
        lambda arg: fn(arg)
        if not hasattr(arg, "__iter__")
        else sum(coeff * arg.parent()(fn(elem)) for elem, coeff in arg)
    )


def extend_bilinear(fn):
    xtl = extend_linear
    return lambda left, right: xtl(lambda l: xtl(lambda r: fn(l, r))(right))(left)
