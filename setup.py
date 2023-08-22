import setuptools
# from setuptools import setup

# setup(
#     name='neumiss',
#     version='0.0.1',
#     author='Marine Le Morvan and Alexandre Perez-Lebel',
#     author_email='alexandre.perez@inria.fr',
#     packages=['neumiss', 'neumiss.test'],
#     scripts=['bin/script1', 'bin/script2'],
#     url='http://pypi.python.org/pypi/PackageName/',
#     license='LICENSE.txt',
#     description='An awesome package that does something',
#     long_description=open('README.txt').read(),
#     install_requires=[
#         "Django >= 1.1.1",
#         "pytest",
#     ],
# )

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