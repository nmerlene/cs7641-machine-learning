# About

CS7641 Assignment #1 - Supervised Learning

# Contents

* [data](./data) - Data directory
* [figures](./figures) - Directory for report figures
* [runs](./runs) - Runs directory
* [suplearn](./suplearn) - Source code directory
* [.gitignore](./.gitignore) - Ignore these files/directories, never track/commit
* [environment.yml](./environment.yml) - Config file used by [conda](https://docs.conda.io/en/latest/) to create a conda environment
* [README.md](./README.md) - You're reading me, hi
* [setup.py](./setup.py) - Script used by [setuptools](https://setuptools.readthedocs.io/en/latest/setuptools.html) to install this directory as a Python package

## To Install

0. Install `conda`: https://docs.conda.io/projects/conda/en/latest/user-guide/install/

1. Create environment:

```bash
conda env create -f environment.yml
```

2. Activate environment:

```bash
conda activate ml
```

## To Run

Main script help menu:

```bash
suplearn -h
```

Ex: Run `DTLearner` on `wine` dataset

```bash
suplearn -l DTLearner -d wine
```
