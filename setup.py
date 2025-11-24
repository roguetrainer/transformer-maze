from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="transformer-maze",
    version="0.1.0",
    author="Ian Forde",
    author_email="",
    description="Understanding Transformers Through Maze Solving: From Sequential to Parallel Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roguetrainer/transformer-maze",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
        "notebook>=7.0.0",
        "ipywidgets>=8.0.0",
        "plotly>=5.14.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "ode": [
            "torchdiffeq>=0.2.3",
        ],
    },
    package_dir={"": "src"},
    include_package_data=True,
)
