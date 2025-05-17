Integrative Analysis and Imputation of Multiple Data Streams via Deep Gaussian Processes
-----

by Ali A. Septiandri, Deyu Ming, F. A. Diaz De la O, Takoua Jendoubi, Samiran Ray

## Abstract

**Motivation:** Healthcare data, particularly in critical care settings, presents three key challenges for analysis. First, physiological measurements come from different sources but are inherently related. Yet, traditional methods often treat each measurement type independently, losing valuable information about their relationships. Second, clinical measurements are collected at irregular intervals, and these sampling times can carry clinical meaning. Finally, the prevalence of missing values. Whilst several imputation methods exist to tackle this common problem, they often fail to address the temporal nature of the data or provide estimates of uncertainty in their predictions.

**Results:** We propose using deep Gaussian process emulation with stochastic imputation, a methodology initially conceived to deal with computationally expensive models and uncertainty quantification, to solve the problem of handling missing values that naturally occur in critical care data. This method leverages longitudinal and cross-sectional information and provides uncertainty estimation for the imputed values. Our evaluation of a clinical dataset shows that the proposed method performs better than conventional methods, such as multiple imputations with chained equations (MICE), last-known value imputation, and individually fitted Gaussian Processes (GPs).

## Installation
`dgpsi` currently requires Python version 3.9. The package can be installed via `pip`:

```bash
pip install dgpsi
```

or `conda`:

```bash
conda install -c conda-forge dgpsi
```

However, to gain the best performance of the package or you are using an Apple Silicon computer, we recommend the following steps for the installation:
* Download and install `Miniforge3` that is compatible to your system from [here](https://github.com/conda-forge/miniforge).
* Run the following command in your terminal app to create a virtual environment called `dgp_si`:

```bash
conda create -n dgp_si python=3.9.13 
```

* Activate and enter the virtual environment:

```bash
conda activate dgp_si
```

* Install `dgpsi`:
    - for Apple Silicon users, you could gain speed-up by switching to Apple's Accelerate framework:

    ```bash
    conda install dgpsi "libblas=*=*accelerate"
    ```

    - for Intel users, you could gain speed-up by switching to MKL:

    ```bash
    conda install dgpsi "libblas=*=*mkl"
    ```

    - otherwise, simply run:
    ```bash
    conda install dgpsi
    ```

## Demo and documentation
Please see [demo](https://github.com/mingdeyu/DGP/tree/master/demo) for some illustrative examples of the method. The API reference 
of the package can be accessed from [https://dgpsi.readthedocs.io](https://dgpsi.readthedocs.io).