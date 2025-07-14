from setuptools import setup, find_packages

setup(
    name="schwan",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "deepxde",
        "numpy",
        "matplotlib",
        "scipy",
    ],
    author="Jan Piotraschke",
    description="Sharing the tacit knowledge of experienced workers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Jan-Piotraschke/Schwanensee",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
