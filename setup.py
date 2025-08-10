import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)

with open('README.md', 'r') as fh:
    long_description = fh.read()

BUILD_CUDA = os.getenv("BUILD_CUDA", "1") == "1"
BUILD_ALLOW_ERRORS = os.getenv("BUILD_ALLOW_ERRORS", "1") == "1"

CUDA_ERROR_MSG = (
    "{}\n\n"
    "Failed to build the CUDA extension due to the error above. "
    "You can still use and it's OK to ignore the error above, although some "
    "post-processing functionality may be limited (which doesn't affect the results in most cases; "
    "(see https://github.com/facebookresearch/segment-anything-2/blob/main/INSTALL.md).\n"
)


setup(
    name='gvm',
    version='0.0.1',
    author='yongtaoge',
    author_email='yongtao.ge@adelaide.edu.au',
    description='Code for Generative Video Matting.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=None,
    packages=find_packages(exclude=('configs', 'docs', 'scripts', 'extensions', 'data', 'requirements'),),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[],
)
