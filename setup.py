# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import setuptools

version = '0.1.0'

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
    packages=["prv_accountant"],
    python_requires=">=3.7.0",
    include_package_data=True,
    extras_require={
        "extra": [
            "plotly",
            "tqdm",
            "jupyter",
            "sympy",
            "tensorflow-privacy",
            "nbconvert",
            "pandas",
            "pytest"
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
