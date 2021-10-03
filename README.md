 Welcome to Parametric Scattering Networks
==============================
                                                              
This repository contains the code for Parameteric Scattering Networks. It also contains code to run and test new hybrid architectures for the small sample regime. 

100 Sample CIFAR-10 Challenge
----------------


When combined in with a wide residual network, our learnable scattering networks define the SOTA for 100 sample CIFAR-10 accuracy. We would like to invite any and all researchers who believe they can improve on our results to try and do so using this repository. To obtain comparable results when subsampling from such a large training set, it is important to use the same seeds and to control for deterministic computations. Our repository does both. By running the ```competition/cifar-10_100sample.py``` script, users can generate our state of the art result on CIFAR-10. The results will automatically be logged to mlflow. By modifying the same script and corresponding code under ```parametricSN/```, users can insert their own architectures.

```
cd /path/to/repository/ParametericScatteringNetworks
pyton competition/cifar-10_100sample.py
```
Explore The Mortlet Wavelet Filters we Optimize
------------

<p align="center">
(left) Filters in the fourier domain (middle) Real part of the filters (right) Imaginary part of the filters
 </p>
 <p align="center">
<img src="gifs/scatteringFilterProgressionFourier500epochs.gif" width="225" height="225">            <img src="gifs/scatteringFilterProgressionReal500epochs.gif" width="225" height="225">                <img src="gifs/scatteringFilterProgressionImag500epochs.gif" width="225" height="225">      
</p>


The above gifs visually depict the optimizaiton of our scattering network's morlet wavelet filters. Each frame corresponds to one batch gradient descent step using a 1000 sample subset of CIFAR-10 for training. For instance, the 30th frame corresponds to the positions of the filters after 30 steps of batch gradient descent. The filters were initialized from a tight-frame.




You can use the following notebook to explore the parameters used to create the filters.

Get Setup
------------

Start by cloning the repository

PIP
-----

```
python3 -m venv parametricSN
source parametricSN/bin/activate
pip install -r ParametricScatteringNetworks/dependencies/pip_requirements.txt
```


Conda
------
Prerequisites
- Anaconda/Miniconda 

To create the `parametricSN` conda environment, enter the following in the command prompt: 
```
conda env create -f dependencies/environment.yml
```
To active the `parametricSN` conda environment, enter the following: 
```
conda activate parametricSN
```
Datasets
------------
Our empirical evaluations are based on three image datasets, illustrated in the Figure below. We subsample each dataset at various sample sizes in order to showcase the performance of scattering-based architectures in the small data regime. CIFAR-10 and [KTH-TIPS2](https://www.csc.kth.se/cvap/databases/kth-tips/credits.html) are natural image and texture recognition datasets (correspondingly). They are often used as general-purpose benchmarks in similar image analysis settings. [COVIDx CRX-2](https://www.kaggle.com/andyczhao/covidx-cxr2) is a dataset of X-ray scans for COVID-19 diagnosis; its use here demonstrates the viability of our parametric scattering approach in practice, e.g., in medical imaging applications.

Experiments
------------
All experiments from Parameteric Scattering Networks can be reproduced using the scripts in the experiments folder. For instance, the following command will run our Scattering+LinearLayer 100 sample CIFAR-10 experiment. 
```
python parametricSN/experiments/cifar/ll/ll_100sample_cifar10_experiment.py
```
Running experiments on the [KTH-TIPS2](https://www.csc.kth.se/cvap/databases/kth-tips/credits.html) dataset can be done similarly.
```
python parametricSN/experiments/kth/ll_kth_sample-experiment.py
```
For [COVIDx CRX-2](https://www.kaggle.com/andyczhao/covidx-cxr2)
```
python parametricSN/experiments/xray/ll/ll_100sample_xray_experiment.py
```
All the results and plots are automatically saved in MLflow. 

Evaluating Robustness to Deformation
--------------
In [previous work](https://www.di.ens.fr/~mallat/papiers/ScatCPAM.pdf)(Section 2.5), the scattering transform has been shown to be robust to small deformations. We have created an experiment to test the empirical validity of this theorem for trained parameteric scattering networks (fixed or learnable). To compare different models, the user must first train them on the dataset of interest (using our scripts under the experiments folder) with the 'model' 'save' variable set to True. Once different models have been trained, comparing their deformantion stability is simple: just provide the paths to the desired model's mlflow artifacts (E.G. ```"/ParametricScatteringNetworks/mlruns/1/03f1f015288f47dc81d0529b23c25bf1/artifacts/model"```) to our deformantion script. In this way, multiple different models can be compared:

```
python parametricSN/evaluate_deformed_representation.py <artifact_path1> <artifact_path2>
```

The resulting figures will automatically be logged to mlflow. Below is an example of the script's output for 4 models using the rotation deformation. The figures show, in order, (x) the image before the deformation, (x_tilde) the image after the deformation (visualized at its most deformed point), and the plot of the defomation strength (x-axis) and the distance betweren s(x) and s(x_tilde) normalized by s(x) where s is the scattering transform (y-axis). The horizontal lines are a baseline that indicate the average distance for 50 images between s(x) and s(random_im) normalized by s(x) where random_im is a randomly chosen image from the dataset.
 



Project Organization
------------

    ├── conf                     <- Configuration folder
    ├── data                     <- Contains datasets - to create the different datasets please see section Datasets
    ├── experiments        
    │   ├── cifar                <- All scripts to reproduce cifar experiments.
    |       ├── cnn              <- Scripts tp run all experiments of hybrid sacttering + cnn.
    |       ├── ll               <- Scripts tp run all experiments of hybrid sacttering + linear layer.
    |       └── onlycnn          <- Scripts tp run all experiments of cnn without scattering priors.
    │   ├── kth                  <- All scripts to reproduce KTH-TPIS2 experiments.
    │   └── xray                 <- All scripts to reproduce Covidx CRX-2 experiments.
    |       ├── cnn              <- Scripts tp run all experiments of hybrid sacttering + cnn.
    |       ├── ll               <- Scripts tp run all experiments of hybrid sacttering + linear layer.
    |       └── onlycnn          <- Scripts tp run all experiments of cnn without scattering priors.
    ├── kymatio                  <- Folder copied from: https://github.com/kymatio/kymatio.
    ├── parametricSN 
    │   ├── data_loading         <- Wrapper for subsampling the cifar-10, KTH-TIPS2 and Covidx CRX-2 based on given input.
    │   ├── models               <- Contains all the  pytorch NN.modules for this project.
    │   └── notebooks            <- Jupyter notebooks.
    │   └── training             <- Contains train and test functions.
    │   └── utils                <- Helpers Functions.
    │   └── main.py              <- Source code.
    ├── Dependencies 
    │   └── environment.yml      <- The conda environment file for reproducing the analysis environment.
    |   └── pip_requirements.yml <- The pip requirements file.
    ├── mlruns                   <- All the experiment results are automatically saved in this folder. 
    
    


