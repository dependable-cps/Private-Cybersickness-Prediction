# Private-Cybersickness-Detection using differerential privacy
This repository accompanies the paper Preserving Personal Space: Differentially Private Cybersickness Detection in Immersive Virtual Reality Environments, under submission ISMAR 2024. The repository contains the main code of DP-enabled private cybersickness deetction . The code is tested on Python 3.10, and GPUs are needed to accelerate deep learning models training in both non-private and private settings and run the membership inference attacks.
![n mmmn ,m](https://github.com/ripankundu/Private-Cybersickness-Prediction/assets/63242071/8eef16cf-2d5f-409e-9781-d9ea3712262d)


## Installation

#### Install Python packages.

```http
  pip install -r requirements.txt
```

## Download the Dataset and pre-processing
1- Simulation 2021 Dataset from the following link [Simulation 2021](https://sites.google.com/view/savelab/research?authuser=0) and Gameplay Dataset from the following link [Gameplay](https://github.com/tmp1986/UFFCSData)

2- Then extract data and pre-process for both the non-private and private deep learning models using the following python script:
```http
  Data_preprocessor_for_models.py
```


## Run the DL models in non-private setting for cybersickness detection
Access to the Folder Non_private_model you will find each of the four models script and you can run the non-private cybersickness detection models using the following script

```http
  Non-private CNN model.py
```

## Run the Memebership inference attack (MIA)
To run the MIA you need to access to the Folder MIA attack and you can run the attack for the LSTM models using the following script

```http
  MIA generation.py
```
For run the attack for the others non-private and private models you need to change only the model architecture. For example, if you want to run the attack for the GRU model you just need to replace the LSTM model structure with GRU model structure.

## Run the DP-enabled private models for cybersickness detection 
Access to the Folder DPSGD-based Private models and again you will find each of the four private DL models script and you can run the private cybersickness detection models using the following script

```http
  Private LSTM model.py
```



