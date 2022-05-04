from setuptools import setup, find_packages


setup(
    name="tensorflow_ensembles",
    version="0.0.1",
    license="MIT",
    author="Caio Davi",
    author_email="caio.davi@tamu.edu",
    packages=find_packages("src"),
    package_dir={"": "src"},
    url="https://github.com/caio-davi/tensorflow_ensembles",
    keywords="tensorflow ensemble deep learning machine-learning PSO FSS Neural Networks",
    install_requires=["tensorflow", "tensorflow_probability"],
)
