import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SAINT",
    version="0.0.1",
    description="SAINT original repo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marineLM/SAINT",
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.8",
    install_requires=[
        'torch',
    ],
)