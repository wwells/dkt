from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name='khan-deepkt',
    version="0.1",
    license="MIT",
    packages=find_packages(),
    python_requires='==3.8.*',
    install_requires=requirements
)