# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import setuptools

version = '0.2.0'

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='prv_accountant',
    version=version,
    description='A fast algorithm to optimally compose privacy guarantees of differentially private (DP) mechanisms to arbitrary accuracy.',  # noqa: E501
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/microsoft/prv_accountant',
    author='Microsoft Corporation',
    packages=["prv_accountant", "prv_accountant.privacy_random_variables"],
    python_requires=">=3.7.0",
    include_package_data=True,
    extras_require={
        "test": [
            "pytest>=7,<10",
            "flake8>=5,<8"
        ],
        "notebooks": [
            "plotly>=5,<7",
            "nbconvert>=7,<8",
            "ipykernel>=6,<8",
            "pandas>=1.5,<3",
            # IPython 8.13 declares Python 3.8 support but requires 3.9 at import.
            "ipython>=8,<8.13; python_version<'3.9'",
            # fastjsonschema 2.22 uses Python 3.10 syntax on older interpreters.
            "fastjsonschema>=2.15,<2.22; python_version<'3.10'"
        ],
        "reference": [
            "dp-accounting==0.4.3"
        ],
        "extra": [
            "plotly",
            "tqdm",
            "jupyter",
            "sympy",
            "tensorflow-privacy",
            "nbconvert",
            "pandas",
            "pytest",
            "flake8"
        ]
    },
    install_requires=[
        "numpy",
        "scipy"
    ],
    scripts=[
        'bin/compute-dp-epsilon',
    ],
    test_suite="tests",
    zip_safe=False
)
