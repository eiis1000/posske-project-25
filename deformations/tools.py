from .config import enable_logging

try:
    from line_profiler import profile
except ImportError:
    profile = lambda *_, **__: lambda fn: fn


def extend_linear(fn):
    return (
        lambda arg: fn(arg)
        if not hasattr(arg, "__iter__")
        else sum(coeff * arg.parent()(fn(elem)) for elem, coeff in arg)
    )


def extend_to_coeffs(fn):
    return (
        lambda arg: fn(arg)
        if not hasattr(arg, "__iter__")
        else sum(fn(coeff) * arg.parent()(elem) for elem, coeff in arg)
    )


def extend_bilinear(fn):
    xtl = extend_linear
    return lambda left, right: xtl(lambda l: xtl(lambda r: fn(l, r))(right))(left)


@profile
def map_elements(input, fn, item_map):
    if isinstance(input, dict):
        input = input.items()
    for k, v in input:
        if hasattr(k, "is_zero") and k.is_zero():
            continue
        mapped_k, mapped_v = item_map(k, v)
        fn((mapped_k, mapped_v))


@profile
def map_collect_elements(input, item_map, zero=0):
    output = {}

    def update(kv):
        k, v = kv
        output[k] = output.get(k, zero) + v

    map_elements(input, update, item_map)

    return output


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


if enable_logging:

    class AccumWrapper:
        def __init__(self, obj):
            print("Making accum")
            self._wrapped = obj

        def __isub__(self, other):
            print(f"-= {other}")
            self._wrapped -= other
            return self

        def __iadd__(self, other):
            print(f"+= {other}")
            self._wrapped += other
            return self

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        def __repr__(self):
            return f"Wrapper({repr(self._wrapped)})"
else:
    AccumWrapper = lambda x: x
