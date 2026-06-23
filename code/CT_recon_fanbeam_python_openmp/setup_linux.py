from setuptools import setup, Extension
import numpy as np
module = Extension(
    "recon",
    sources=["recon.c"],
    extra_compile_args=["-fopenmp"],  # Linux/macOS에서는 -fopenmp 사용
    extra_link_args=["-fopenmp"],     # 링크 플래그도 추가해야 함
    include_dirs=[np.get_include()],
)

setup(
    name="recon",
    version="1.0",
    description="Forward, Backward projection C-extension module",
    ext_modules=[module],
)