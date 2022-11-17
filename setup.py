from setuptools import find_packages, setup


python_requires = "~=3.7"
install_requires = [
    "torchsde",
    "jax",
    "equinox",
    "diffrax",
]

setup(
    name="fractional_neural_sde",
    packages=find_packages(
        include=["fractional_neural_sde", "fractional_neural_sde_jax"]
    ),
    python_requires=python_requires,
    install_requires=install_requires,
)
