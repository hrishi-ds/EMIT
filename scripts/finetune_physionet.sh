# !/bin/bash

cd ..
python -m src.finetuning.finetune_physionet epochs=1 pt_model='EMIT_PHYSIONET_PT_lr_0.0005_err_coef_3_mask_threshold_0.01_insig_prob_0.9' repeats=1 batch_size=16
