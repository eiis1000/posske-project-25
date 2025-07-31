from .config import enable_logging


def extend_linear(fn):
    return (
        lambda arg: fn(arg)
        if not hasattr(arg, "__iter__")
        else sum(coeff * arg.parent()(fn(elem)) for elem, coeff in arg)
    )


def extend_bilinear(fn):
    xtl = extend_linear
    return lambda left, right: xtl(lambda l: xtl(lambda r: fn(l, r))(right))(left)


def comm(a, b):
    return a.bracket(b) if hasattr(a, "bracket") else a * b - b * a


def anticomm(a, b):
    return a * b + b * a


def compose(*f):
    ret = (lambda *x: compose(*f[:-1])(f[-1](*x))) if f else lambda x: x
    ret.__name__ = "(" + " o ".join(func_name(fn) for fn in f) + ")"
    ret.__qualname__ = ret.__name__
    return ret


def func_name(fn):
    if hasattr(fn, "__func__"):
        return func_name(fn.__func__)
    # elif hasattr(fn, "__qualname__"):
    #     return fn.__qualname__
    elif hasattr(fn, "__name__"):
        return fn.__name__
    elif hasattr(fn, "__repr__"):
        return fn.__repr__()
    else:
        return str(fn)


def wrap_logging(fn):
    if not enable_logging:
        return fn

    def wrapper(*args, **kwargs):
        fname = func_name(fn)
        x = fn(*args, **kwargs)
        x_str = str(x)
        if len(x_str) > 10:
            x_str = "\n" + (" " * 12) + "= " + x_str
        else:
            x_str = " = " + x_str
        print(f"    logging: {fname}({args}){x_str}")
        return x

    return wrapper
