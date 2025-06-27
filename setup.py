from setuptools import setup, find_packages

setup(
    name='hft_rl_arbitrage',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)