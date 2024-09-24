# !/bin/bash

cd ..
python -m src.pretraining.pretrain_mimic epochs=1 samples_per_epoch=100 batch_size=16 mask_threshold=0.001 insignificant_prob=0.9
