# EMIT (Event-Based Masked Auto-Encoding for Irregular Time Series)

Code for EMIT paper (ICDM 2024) <br>
Title :  Event-Based Masked Auto-Encoding for Irregular Time Series <br>
Paper link : <font color="red">(PLACEHOLDER ARXIV LINK)</font> <br>
Authors : Hrishikesh Patel, Ruihong Qiu, Adam Irwin, Shazia Sadiq, Sen Wang

## Overview
EMIT is a framework designed for event-based masked auto-encoding applied to irregular time series data. This project aims to improve model pretraining using event masking techniques. The framework includes scripts for preprocessing, event mask generation, and model training on various datasets, including MIMIC and PHYSIONET.

### Environment Setup
Clone the repository and then run the following in terminal.
```
cd EMIT
conda env create -f environment.yml
conda activate emit
```

### Dataset
* Download PhysioNet2012 dataset from https://physionet.org/content/challenge-2012/1.0.0/. (available to anyone)
* Download MIMIC-III from https://physionet.org/content/mimiciii/1.4/, (credentialed access)

Unzip these files into the directory ```EMIT/data/raw/```. 

### Preprocess Raw Data

```
cd scripts
bash preprocess_physionet.sh
```
Note : If you encounter data path related error, kindly check the variable `raw_data_path` in the file `preprocess_physionet.py` and adjust according to your data path.

### Generate Pretraining Data
```
cd scripts
bash get_pretraining_data_physionet.sh
```
### Generate Event Masks 
```
cd scripts
bash get_event_masks_physionet.sh
```
Note : You may consider changing deafult variables (<i>rate of change threshold</i> & <i>insignificant probability</i>) in the `get_event_masks_physionet.sh` script.

### Pretrain Model
```
cd scripts
bash pretrain_physionet.sh
```
* The pretraining configurations can be modified from the `config/pt_config_physionet.yaml` file. 
* Pretrained models are located in the pretrained_models/ directory. You can load these models for evaluation or further training.

### Finetune Model
```
cd scripts
bash finetune_physionet.sh
```
The finetuning configurations can be modified from the `config/ft_config_physionet.yaml` file.


### Results
Results from model finetuning can be found in the `results/` directory.

### Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.

### License
This project is licensed under the MIT License - see the LICENSE file for details.






