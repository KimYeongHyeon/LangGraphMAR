import os
from setuptools import Extension, setup

llvm_path = "/opt/homebrew/opt/llvm"

module = Extension(
    "recon",
    sources=["recon.c"],
    extra_compile_args=[
        "-Xpreprocessor", "-fopenmp",
        f"-I{llvm_path}/include",  # Include LLVM headers
        "-I/opt/homebrew/Caskroom/miniconda/base/envs/ctmar/lib/python3.11/site-packages/numpy/core/include"
    ],
    extra_link_args=[
        "-lomp",  # Link against the OpenMP library
        f"-L{llvm_path}/lib",  # LLVM libraries path
    ],
      include_dirs=[
      "/opt/homebrew/opt/libomp/include",  # Correct path to omp.h
      "/opt/homebrew/Caskroom/miniconda/base/envs/ctmar/lib/python3.11/site-packages/numpy/core/include",
      ]
)

setup(
    name="recon",
    ext_modules=[module],
)

'''
For macOS, the default clang compiler does not support OpenMP out of the box. You need to either install a version of clang that supports OpenMP (typically through brew install llvm) or use gcc, which supports OpenMP.
'''