# Happywhale  (Kaggle challenge)
Author: Leandro Salemi <br>
Email: <salemileandro@gmail.com>

Github repository for the Kaggle challenge [Happywhale - Whale and Dolphin Identification
](https://www.kaggle.com/c/happy-whale-and-dolphin/overview)

Conda environment can be reproduced by 

    conda env create -n happywhale -f environment.yaml
    conda activate happywhale

## Scripts
    - python3 scripts/classification/train_classifier.py


## To Do
The goal is to identify the individuals not the species.
However if two pictures are of different species, then they must be of different
individuals too

    - Step 1: classification problem
        * Take care of class imbalance (resampling, class weighting, class aggregation)
        * Use transfer learning with pre-trained + finetuning
    - Step 2: "fin" recognition
        * Siamese CNN network ? Backbone ? To be explored further