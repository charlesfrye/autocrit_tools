import setuptools

version = "0.0.1"

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="autocrit_tools",
    version="0.0.1",
    author="Charles Frye",
    author_email="cfrye59@gmail.com",
    description="Scripts and utilities for working with autocrit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent"],
)
