from setuptools import setup, find_packages

setup(
    name="CoCoNut",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pynvml",
        "torch",
    ],
    author="HBigo",
    author_email="hbigopk@gmail.com",
    description="CoCoNut",
)
