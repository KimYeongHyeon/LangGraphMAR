"""Build the opt-in fast projector: python setup_fast.py build_ext --inplace  (then copy recon_fast*.so into code/).

-ffp-contract=off / -mno-fma are required: a fused multiply-add rounds once instead of twice and would change the result.
AVX2 is used only for element-wise IEEE operations, which round exactly like the scalar code. Needs an AVX2 CPU.
"""
from setuptools import setup, Extension
import numpy as np

module = Extension(
    "recon_fast",
    sources=["recon_fast.c"],
    extra_compile_args=["-fopenmp", "-O3", "-mavx2", "-mno-fma", "-ffp-contract=off"],
    extra_link_args=["-fopenmp"],
    include_dirs=[np.get_include()],
)
setup(name="recon_fast", version="1.0", description="Order-preserving fast FP/BP for LangGraph-MAR", ext_modules=[module])
