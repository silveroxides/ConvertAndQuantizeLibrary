"""
Setup script for backwards compatibility.
This script reads configuration from pyproject.toml.
"""

from setuptools import setup, find_packages

setup(
    packages=find_packages(exclude=["tests", "examples", "docs"]),
)
