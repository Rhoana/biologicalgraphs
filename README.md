## Skeleton Benchmark

The skeleton benchmark and code for the 2019 CVPR paper *Biologically-Constrained Graphs for Global Connectomics Reconstruction* [1]. For more information: https://www.rhoana.org/biologicalgraphs.

````
git clone https://github.com/Rhoana/biologicalgraphs.git .
cd biologicalgraphs
git checkout skeleton_benchmark
conda create -n biographs_env python=2.7
conda install --file requirements.txt
````

```` 
cd ../../skeletonization
python setup.py build_ext --inplace
cd ../transforms
python setup.py build_ext --inplace
````

Add the parent directory to this repository to your PYTHONPATH variable. 

### Example Script

The example scripts are in `neuronseg/scripts`. These scripts run different skeleton generation strategies on the benchmark dataset.

### Dataset

We manually identified skeleton endpoints in the ground truth of one half of the Kasthuri dataset [2].

### Citations
    
[1] Matejek, B., Haehn, D., Zhu, H., Wei, D., Parag, T. and Pfister, H., 2019. Biologically-Constrained Graphs for Global Connectomics Reconstruction. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ (pp. 2089-2098).
    
[2] Kasthuri, N., Hayworth, K.J., Berger, D.R., Schalek, R.L., Conchello, J.A., Knowles-Barley, S., Lee, D., Vázquez-Reina, A., Kaynig, V., Jones, T.R. and Roberts, M., 2015. Saturated reconstruction of a volume of neocortex. _Cell_, _162_(3), pp.648-661.