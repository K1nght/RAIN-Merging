#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rain-merging",
    version="0.1.0",
    author="RAIN-Merging Contributors",
    author_email="",
    description="Null-space Projection Model Merging Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/RAIN-Merging",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "optimized": [
            "flash-attn>=2.0.0",
            "bitsandbytes>=0.41.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rain-merge=pipeline:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
