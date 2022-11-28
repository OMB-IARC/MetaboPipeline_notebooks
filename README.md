# Notebooks for Metabopipeline

This repository contains an ensemble of notebooks to apply a "data science" approach to the metabolomics data outputs from the MetaboPipeline_bioinfo pipeline or from the vendor softwares (e.g., Agilent MassHunter). These notebooks allow a number of statistical and machine learning methods to be applied to analyze the metabolomics datasets, including metadata from the EPIC cohort studies.


## Available notebooks

In the <code>notebooks</code> subfolder are located Jupyter notebooks applied to four public datasets and one synthetic dataset.

Each subfolder with name *analyse_XXX* contains one notebook with an example of visualisations, statitical tests, analysis that can be applied to a dataset. These notebooks do not contain all available methods. An exhaustive presentation is available in the <code>O-TUTORIAL</code> subfolder, in which each notebook present a step for metabolomics data analysis.


## Integration in the MetaboPipeline_bioinfo pipeline

This set of notebooks is designed to be used in the MetaboPipeline_bioinfo pipeline. Package management and version control are handled with a Docker image ([link to DockerHub](https://hub.docker.com/r/maxvin/data_science_img)).

For for information about installation, please see [MetaboPipeline_bioinfo pipeline repo](https://github.com/OMB-IARC/MetaboPipeline_bioinfo).


---

## Binder (still in development)

To launch each notebook, just click on the button next to the notebook name :
- *Binder integration of notebooks currently in development*


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maxvincent24/metabopipeline_notebooks/HEAD?labpath=notebooks%2F0-TUTORIAL%2F1-explore_data.ipynb)
