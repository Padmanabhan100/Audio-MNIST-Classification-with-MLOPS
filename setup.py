from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="Padmanabhan",
    description="Audio MNIST With MLOps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Padmanabhan100/Audio-MNIST-Classification-with-MLOPS",
    author_email="paddyrolex10@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        'dvc',
        'pandas',
        'scikit-learn',
        'librosa',
        'tensorflow',
        'seaborn',
        'matplotlib',
        'numpy'
    ]
)