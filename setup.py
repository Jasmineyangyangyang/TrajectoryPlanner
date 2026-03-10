from setuptools import setup, find_packages

setup(
    name="trajectory_planner",
    version="0.1.0",
    description="Quintic polynomial trajectory planner and Stanley controller",
    author="YANG JIAXIN",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
)
