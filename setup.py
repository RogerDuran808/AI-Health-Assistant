# setup.py
from setuptools import setup, find_packages

setup(
    name="ai_health_assistant",              # nom del paquet
    version="0.1.0",
    packages=find_packages(where="src"),     # troba només dins src/
    package_dir={"": "src"},                 # indica que src/ és el directori arrel de paquets
    install_requires=[                       # dependències mínimes
        "pandas",
        "numpy",
        "scikit-learn",
        "imbalanced-learn"
    ],
    python_requires=">=3.8",
)
