# Happywhale  (Kaggle challenge)
Author: Leandro Salemi <br>
Email: <salemileandro@gmail.com>

Github repository for the Kaggle challenge 
[Happywhale - Whale and Dolphin Identification](https://www.kaggle.com/c/happy-whale-and-dolphin/overview).
The data can be downloaded and using the Kaggle API

    kaggle competitions download -c happy-whale-and-dolphin

The environment variable `KAGGLE_USERNAME` and `KAGGLE_KEY` must be set before (see Kaggle API).


Once the zip file is downloaded, the following commands can be executed to set up properly 
the directory tree

    mkdir -p input/happy-whale-and-dolphin
    mv happy-whale-and-dolphin.zip input/happy-whale-and-dolphin
    cd input/happy-whale-and-dolphin
    unzip -q happy-whale-and-dolphin.zip

The zip file is quite voluminous, it may take few minutes to extract the data. The file `happy-whale-and-dolphin.zip`
can be discarded after it has been extracted.


## Conda Environment
    


Conda environment can be reproduced by 

    conda env create -n happywhale -f environment.yaml

Execute `conda activate happywhale` to activate the environment.
    

## Scripts
    python3 scripts/classification/train_classifier.py


## Experiment tracking
Experiment and performance tracking done via 
[MLflow](https://mlflow.org/docs/latest/index.html) and can be monitored with the mlflow UI

    mlflow ui



## To Do
The goal is to identify the individuals not the species.
However if two pictures are of different species, then they must be of different
individuals too

    - Step 0: 
        * [v] Explore the data 
    - Step 1: classification problem
        * [v] Take care of class imbalance (resampling, class weighting, class aggregation)
        * [v] Write training script
        * [x] Explore the parameter space
        * [x] Write script on the preprocessing to clean-up
        * [x] Write script for evaluation (multi-class confusion matrix !)
    - Step 2: "fin" recognition
        * [x] Siamese CNN network ? Backbone ? To be explored further