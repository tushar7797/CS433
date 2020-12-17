import setuptools

setuptools.setup(
    name="latento",
    version="0.0.1",
    author="Wouter Saelens",
    author_email="wouter.saelens@gmail.com",
    description="Using pyro and torch for single-cell data",
    long_description="This package contains helper functions to make working with pyro and torch easier focused on single-cell data. These include distributions commonly found in single-cell data and linker functions to get the data into scanpy and anndata",
    long_description_content_type="text/markdown",
    url="https://github.com/probabilistic-cell/latento",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = [
        "torch",
        "pyro-ppl",
        "tqdm",
        "scanpy",
        "anndata",
        "pandas",
        "numpy",
        "altair",
        "matplotlib",
        "seaborn",
        "numpyro"
    ]
)
