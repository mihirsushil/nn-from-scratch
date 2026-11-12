"""Build the optional C++ kernels: python setup.py build_ext --inplace."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "nn_cpp",
        ["csrc/nn_cpp.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="nn-from-scratch-cpp",
    version="0.1.0",
    description="C++ forward kernels for the from-scratch autodiff engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
