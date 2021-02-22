"""Package Setup"""

import setuptools

setuptools.setup(
    name="suplearn",
    description="Supervised Learning",
    packages=setuptools.find_packages(),
    python_requires=">=3.7, <4",
    install_requires=["scikit-learn", "mlxtend", "pandas", "numpy", "matplotlib", "seaborn"],
    entry_points={"console_scripts": ["suplearn=suplearn.main:main"]},
)
