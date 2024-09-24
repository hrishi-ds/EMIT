# !/bin/bash

cd ..
python -m src.finetuning.finetune_mimic epochs=1 pt_model='EMIT_MIMIC_PT_lr_0.0005_err_coef_8_mask_threshold_0.001_insig_prob_0.9' repeats=1 batch_size=16
