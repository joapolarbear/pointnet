from setuptools import setup, find_packages
setup(
    name = "pointnet",
    version = "0.1",
    packages = find_packages(),
    install_requires = ["numpy>=1.9.1", "pykdtree>=0.2", "algebra"],
    extras_require = {
        'LAS':  ["laspy"],
        'numba': ["numba"],
    },
    scripts = ['masb.py']
)