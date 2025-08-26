from itertools import chain

from .config import enable_logging

try:
    from line_profiler import profile
except ImportError:
    profile = lambda *_, **__: lambda fn: fn


def extend_linear(fn):
    return (
        lambda arg: fn(arg)
        if getattr(arg, "__iter__", None) is None
        else sum(coeff * arg.parent()(fn(elem)) for elem, coeff in arg)
    )


def extend_to_coeffs(fn):
    return (
        lambda arg: fn(arg)
        if getattr(arg, "__iter__", None) is None
        else sum(fn(coeff) * arg.parent()(elem) for elem, coeff in arg)
    )


def extend_bilinear(fn):
    xtl = extend_linear
    return lambda left, right: xtl(lambda l: xtl(lambda r: fn(l, r))(right))(left)


@profile
def map_elements(input, fn, item_map):
    if type(input) is dict:
        input = input.items()
    for item in input:
        mapped_k, mapped_v = item_map(*item)
        if is_zero(mapped_v):
            continue
        fn((mapped_k, mapped_v))


def _map_collect_elements_update_proto(kv, output):
    k, v = kv
    l = output.get(k, None)
    if l is None:
        output[k] = [v]
    else:
        l.append(v)


@profile
def map_collect_elements(input, item_map, zero=0):
    output_lists = {}

    map_elements(
        input,
        lambda kv: _map_collect_elements_update_proto(kv, output_lists),
        item_map,
    )

    output = {}
    for k, vlist in output_lists.items():
        output[k] = zero.parent().sum(vlist)

    return output


@profile
def sum_dicts(*dicts, zero=0):
    chained = chain.from_iterable(d.items() for d in dicts)
    return map_collect_elements(chained, lambda k, v: (k, v), zero=zero)


def is_zero(input):
    """
    Returns True if the input is zero; if the input is a tuple, returns True if
    ANY of the elements are zero, as we assume it will be used in a product.
    """
    if (iz := getattr(input, "is_zero", None)) is not None:
        return iz()
    elif getattr(input, "dtype", None) is not None:
        # this means we're a numpy array
        return False
    return input == 0


def compose(*f):
    ret = (lambda *x: compose(*f[:-1])(f[-1](*x))) if f else lambda x: x
    ret.__name__ = "(" + " o ".join(func_name(fn) for fn in f) + ")"
    ret.__qualname__ = ret.__name__
    return ret


def func_name(fn):
    # we'll use hasattr because we don't care about speed
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
