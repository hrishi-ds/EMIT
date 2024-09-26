# EMIT (Event-Based Masked Auto-Encoding for Irregular Time Series)

Code for EMIT paper (ICDM 2024)

**Title:** Event-Based Masked Auto-Encoding for Irregular Time Series

**Paper link:** https://arxiv.org/abs/2409.16554

**Authors:** Hrishikesh Patel, Ruihong Qiu, Adam Irwin, Shazia Sadiq, Sen Wang


## Overview
EMIT is a framework designed for event-based masked auto-encoding applied to irregular time series data. This project aims to improve model pretraining using event masking techniques. The framework includes scripts for preprocessing, event mask generation, and model training on various datasets, including MIMIC and PHYSIONET.

## Codebase Structure
```
emit
├── .gitignore
├── LICENSE
├── README.md
├── config
│   ├── ft_config_mimic.yaml
│   ├── ft_config_physionet.yaml
│   ├── pt_config_mimic.yaml
│   └── pt_config_physionet.yaml
├── data
│   ├── .gitkeep
│   ├── pre_processed
│   │   └── .gitkeep
│   ├── pre_training
│   │   └── .gitkeep
│   └── raw
│       └── .gitkeep
├── environment.yml
├── pretrained_models
│   ├── MIMIC
│   │   └── EMIT_MIMIC_PT_lr_0.0005_err_coef_8_mask_threshold_0.001_insig_prob_0.4.h5
│   └── PHYSIONET
│       └── EMIT_PHYSIONET_PT_lr_0.0005_err_coef_3_mask_threshold_0.01_insig_prob_0.7.h5
├── results
│   ├── MIMIC
│   │   └── EMIT_MIMIC_PT_lr_0.0005_err_coef_8_mask_threshold_0.001_insig_prob_0.4_FT_batchsize32_dropout0.4_lr5e-05_weight_decay0.0001.pkl
│   ├── PHYSIONET
│   │   └── EMIT_PHYSIONET_PT_lr_0.0005_err_coef_3_mask_threshold_0.01_insig_prob_0.7_FT_batchsize32_dropout0.4_lr5e-05_weight_decay0.0001.pkl
│   └── results_notebook.ipynb
├── scripts
│   ├── finetune_mimic.sh
│   ├── finetune_physionet.sh
│   ├── get_event_masks_mimic.sh
│   ├── get_event_masks_physionet.sh
│   ├── get_pretraining_data_mimic.sh
│   ├── get_pretraining_data_physionet.sh
│   ├── preprocess_mimic.sh
│   ├── preprocess_physionet.sh
│   ├── pretrain_mimic.sh
│   └── pretrain_physionet.sh
└── src
    ├── __init__.py
    ├── data_preprocessing
    │   ├── preprocess_mimic_iii.py
    │   └── preprocess_physionet_2012.py
    ├── event_masks
    │   ├── generate_event_masks_mimic_iii.py
    │   └── generate_event_masks_physionet_2012.py
    ├── finetuning
    │   ├── __pycache__
    │   │   ├── finetune_mimic.cpython-310.pyc
    │   │   └── finetune_physionet.cpython-310.pyc
    │   ├── finetune_mimic.py
    │   └── finetune_physionet.py
    ├── model.py
    ├── pretraining
    │   ├── __pycache__
    │   │   ├── pretrain_mimic.cpython-310.pyc
    │   │   └── pretrain_physionet.cpython-310.pyc
    │   ├── pretrain_mimic.py
    │   └── pretrain_physionet.py
    └── pretraining_data_preparation
        ├── get_pretraining_data_mimic_iii.py
        └── get_pretraining_data_physionet_2012.py

```

## Usage

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

## Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.

## Cite
If you find this repo useful, please cite

```
@article{EMIT,
  author       = {Hrishikesh Patel and
                  Ruihong Qiu and
                  Adam Irwin and
                  Shazia Sadiq and
                  Sen Wang},
  title        = {Event-Based Masked Auto-Encoding for Irregular Time Series},
  journal      = {CoRR},
  volume       = {abs/2409.16554},
  year         = {2024}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.






