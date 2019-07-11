## Biologically-Constrained Graphs

The code repository for the 2019 CVPR paper *Biologically-Constrained Graphs for Global Connectomics Reconstruction* [1]. For more information: https://www.rhoana.org/biologicalgraphs.

### Dependencies

This code requires the C++ Graph library from Bjoern Andres: http://www.andres.sc/graph.html [2, 3]. This package should not require additional packages or installation as all functions are included in header files.

### Installation

````
git clone https://github.com/Rhoana/biologicalgraphs.git .
cd biologicalgraphs
conda create -n biographs_env python=2.7
conda install --file requirements.txt
````

Change the variable `graph_software_dir` to be the parent directory where you installed the Andres graph repository.

```` 
cd algorithms
python setup.py build_ext --inplace
cd ../evaluation
python setup.py build_ext --inplace
cd ../graphs/biological
python setup.py build_ext --inplace
cd ../../skeletonization
python setup.py build_ext --inplace
cd ../transforms
python setup.py build_ext --inplace
````

Add the parent directory to this repository to your PYTHONPATH variable. 

### Meta Files

Each new dataset needs a meta file named meta/{PREFIX}.meta where {PREFIX} is a unique identifier for the dataset. All functions in this repository require as input this {PREFIX} identifier to find the locations for the requisite datasets (i.e., image, affinities, segmentation, etc.). An example meta file is provided in `neuronseg/meta/Kasthuri-test.meta`. This file contains all of the necessary dataset references. 

### Directory Structure

This python package assumes a certain directory structure. Call the parent directory `{PARENT_DIR}`. All input segmentations should reside in `{PARENT_DIR}/segmentations`. All meta files must be saved in `{PARENT_DIR}/meta`. 

### Example Script

There is an example script to run the complete framework in the `neuronseg/scripts` folder. This script runs the entire framework on the testing portion of the Kasthuri dataset [4]. The ground truth for this dataset is in `neuronseg/golds` and our input segmentations are in `neuronseg/segmentations`. The meta file `neuronseg/meta/Kasthuri-test.meta` contains the relevant links and can act as a guide for future datasets. The network architectures used for the results in the paper are included in the subdirectories in `neuronseg/architectures`. 


### Citations
    
[1] Matejek, B., Haehn, D., Zhu, H., Wei, D., Parag, T. and Pfister, H., 2019. Biologically-Constrained Graphs for Global Connectomics Reconstruction. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ (pp. 2089-2098).

    
[2] Keuper, M., Levinkov, E., Bonneel, N., Lavoué, G., Brox, T. and Andres, B., 2015. Efficient decomposition of image and mesh graphs by lifted multicuts. In _Proceedings of the IEEE International Conference on Computer Vision_ (pp. 1751-1759).

    
[3] Kernighan, B.W. and Lin, S., 1970. An efficient heuristic procedure for partitioning graphs. _Bell system technical journal_, _49_(2), pp.291-307.

    
[4] Kasthuri, N., Hayworth, K.J., Berger, D.R., Schalek, R.L., Conchello, J.A., Knowles-Barley, S., Lee, D., Vázquez-Reina, A., Kaynig, V., Jones, T.R. and Roberts, M., 2015. Saturated reconstruction of a volume of neocortex. _Cell_, _162_(3), pp.648-661.