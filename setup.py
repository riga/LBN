# coding: utf-8


import os
import setuptools


this_dir = os.path.dirname(os.path.abspath(__file__))

keywords = [
    "neural network", "lorentz", "lorentz transformation", "lorentz boost", "autonomous", "feature",
    "feature engineering", "autonomous engineering", "hep", "four momenta", "four vectors",
]

classifiers = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Development Status :: 4 - Beta",
    "Operating System :: OS Independent",
    "License :: OSI Approved :: BSD License",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Information Technology",
]

# read the readme file
with open(os.path.join(this_dir, "README.md"), "r") as f:
    long_description = f.read()

# load installation requirements
with open(os.path.join(this_dir, "requirements.txt"), "r") as f:
    install_requires = [line.strip() for line in f.readlines() if line.strip()]

# load package infos
pkg = {}
with open(os.path.join(this_dir, "lbn.py"), "r") as f:
    # read until lines up to the __version__ info
    lines = f.readlines()
    end = [i for i, l in enumerate(lines) if l.startswith("__version__")][0] + 1
    exec("".join(lines[:end]), pkg)

setuptools.setup(
    name="lbn",
    version=pkg["__version__"],
    author=pkg["__author__"],
    author_email=pkg["__email__"],
    description=pkg["__doc__"].strip().split("\n")[0].strip(),
    license=pkg["__license__"],
    url=pkg["__contact__"],
    keywords=keywords,
    classifiers=classifiers,
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    python_requires=">=2.7",
    zip_safe=False,
    py_modules=["lbn"],
    data_files=[(".", ["LICENSE", "requirements.txt", "README.md"])],
)
