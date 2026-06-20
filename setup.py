#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for StructuraMiner package.
Allows installation via pip and upload to PyPI.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from structuraminer.py
version_file = this_directory / "structuraminer.py"
version = None
for line in version_file.read_text().splitlines():
    if line.startswith("VERSION"):
        version = line.split('=')[1].strip().strip('"')
        break

if version is None:
    version = "1.0.0"  # fallback

setup(
    name="structuraminer",
    version=version,
    author="Love Kaushik",
    author_email="lovekaushik0512.com",
    description="Exhaustive Structural Feature Extraction from Protein Data Bank (PDB) Files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lovekaushik899/StructuraMiner",
    project_urls={
        "Bug Tracker": "https://github.com/lovekaushik899/StructuraMiner/issues",
        "Source Code": "https://github.com/lovekaushik899/StructuraMiner",
        "Documentation": "https://github.com/lovekaushik899/StructuraMiner#readme",
    },
    license="MIT",
    py_modules=["structuraminer"],
    entry_points={
        "console_scripts": [
            "structuraminer=structuraminer:main",
        ],
    },
    python_requires=">=3.8",
    install_requires=[
        "biopython>=1.85",
        "numpy>=2.0",
        "pandas>=2.2",
        "scipy>=1.14",
        "networkx>=3.4",
        "freesasa>=2.2",
        "pydssp>=0.9",
        "tqdm>=4.67",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Multimedia :: Graphics :: 3D Molecular Visualization",
    ],
    keywords=[
        "protein",
        "pdb",
        "bioinformatics",
        "structural-biology",
        "feature-extraction",
        "machine-learning",
        "molecular-dynamics",
        "computational-chemistry",
    ],
    include_package_data=True,
    zip_safe=False,
)
