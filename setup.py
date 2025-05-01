from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    README = fh.read()

setup(
    name="Horama",
    version="0.2.0",
    description="Personal toolbox for experimenting with Feature Visualization",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Thomas FEL, Thibaut BOISSIN, Victor BOUTIN, Agustin PICARD, Paul NOVELLO",
    author_email="thomas_fel@brown.edu",
    license="MIT",
    install_requires=['numpy', 'matplotlib', 'torch', 'torchvision'],
    packages=find_packages(),
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
