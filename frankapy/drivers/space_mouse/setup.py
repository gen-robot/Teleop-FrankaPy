#!/usr/bin/env python
from setuptools import setup, find_packages

__author__ = "Bingwen Wei"
__copyright__ = "2025, Tsinghua University"

install_requires = ["hidapi", "pynput"]

setup(
    name="space_mouse_wrapper",
    author="Bingwen Wei",
    version=1.0,
    packages=["space_mouse_wrapper"],
    package_dir={"": "python"},
    install_requires=install_requires,
    zip_safe=False,
)
