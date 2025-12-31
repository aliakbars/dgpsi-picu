Integrative Analysis and Imputation of Multiple Data Streams via Deep Gaussian Processes
-----

by Ali A. Septiandri, Deyu Ming, F. A. Diaz De la O, Takoua Jendoubi, Samiran Ray

## Abstract

**Motivation:** Healthcare data, particularly in critical care settings, presents three key challenges for analysis. First, physiological measurements come from different sources but are inherently related. Yet, traditional methods often treat each measurement type independently, losing valuable information about their relationships. Second, clinical measurements are collected at irregular intervals, and these sampling times can carry clinical meaning. Finally, the prevalence of missing values. Whilst several imputation methods exist to tackle this common problem, they often fail to address the temporal nature of the data or provide estimates of uncertainty in their predictions.

**Results:** We propose using deep Gaussian process emulation with stochastic imputation, a methodology initially conceived to deal with computationally expensive models and uncertainty quantification, to solve the problem of handling missing values that naturally occur in critical care data. This method leverages longitudinal and cross-sectional information and provides uncertainty estimation for the imputed values. Our evaluation of a clinical dataset shows that the proposed method performs better than conventional methods, such as multiple imputations with chained equations (MICE), last-known value imputation, and individually fitted Gaussian Processes (GPs).

## Installation

If you have not installed the `uv` binary yet, use the standaloner installer:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
then run
```bash
uv sync
```
to install all dependencies.

Note that to run the `dgpsi-dgp-*.ipynb`, you will need a different version of `dgpsi` that can be installed by following these [instructions](https://github.com/mingdeyu/DGP/tree/v2.0-beta2).

## Citing This Work

```
@article{10.1093/bioadv/vbaf305,
    author = {Septiandri, Ali A and Ming, Deyu and DiazDelaO, F A and Jendoubi, Takoua and Ray, Samiran},
    title = {Integrative Analysis and Imputation of Multiple Data Streams via Deep Gaussian Processes},
    journal = {Bioinformatics Advances},
    pages = {vbaf305},
    year = {2025},
    month = {11},
    issn = {2635-0041},
    doi = {10.1093/bioadv/vbaf305},
    url = {https://doi.org/10.1093/bioadv/vbaf305},
    eprint = {https://academic.oup.com/bioinformaticsadvances/advance-article-pdf/doi/10.1093/bioadv/vbaf305/65575896/vbaf305.pdf},
}
```