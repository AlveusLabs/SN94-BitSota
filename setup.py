import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="distributed-automl",
    version="0.0.1",
    description="A distributed AutoML package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hivetensor/AutoMLInfinite",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
