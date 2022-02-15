from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="AlgoTradingDeepLearning",
    version="0.0.1",
    author="Zach Holmes",
    author_email="zholmes13@gmail.com",
    description="AlgoTradingDeepLearning is a deep learning framework for algorithmic trading.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zholmes13/algo_trading_deep_learning",
    license="MIT",
    packages=find_packages()
)
