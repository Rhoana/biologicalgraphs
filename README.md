# **Under Construction!**

## Biologically-Constrained Graphs

The code repository for the 2019 CVPR paper *Biologically-Constrained Graphs for Global Connectomics Reconstruction*. For more information: https://www.rhoana.org/biologicalgraphs.

### Dependencies

This code requires the C++ Graph library from Bjoern Andres: http://www.andres.sc/graph.html. 

### Directory Structure

This python package assumes a certain directory structure. Call the parent directory `{PARENT_DIR}`. The following subdirectories are required:

```{PARENT_DIR}/architectures```
```{PARENT_DIR}/cache```
```{PARENT_DIR}/features/biological```
```{PARENT_DIR}/meta```
```{PARENT_DIR}/segments```
```{PARENT_DIR}/skeletons```

### Installation

```conda create -n biographs_env python=2.7```

### Input Segmentation



### Meta Files

Each new dataset needs a meta file named meta/{PREFIX}.meta where {PREFIX} is a unique identifier for the dataset. All functions in this repository require as input this {PREFIX} identifier to find the locations for the requisite datasets (i.e., image, affinities, segmentation, etc.). 

### Example Script

There is an example script to run the complete framework in the `example/scripts` folder. This will run all aspects of the pipeline on the SNEMI3D dataset. 
