from .config import use_sage_threads
import os
import sage.all as sa

if use_sage_threads:
    os.environ["SAGE_NUM_THREADS"] = "4"

exec(
    """
from sage.all import *
""",
    sa.__dict__,
)


# from sage.structure.indexed_generators import IndexedGenerators
# from sage.algebras.lie_algebras.lie_algebra import LieAlgebraWithGenerators
# from sage.algebras.lie_algebras.lie_algebra_element import LieAlgebraElement
