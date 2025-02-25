from setuptools import find_packages
from distutils.core import setup

setup(
    name="wheel_legged_gym",
    version="1.0.0",
    author="4yang",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="liusiyang.cn@gmail.com",
    description="Isaac Gym environments for wheeled_bipedal robot",
    install_requires=[
        "isaacgym",
        "matplotlib",
        "tensorboard==2.12.0",
        "setuptools==59.5.0",
        "numpy==1.23.5",
        "GitPython",
        "onnx",
    ],
)
