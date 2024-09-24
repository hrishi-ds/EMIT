# !/bin/bash

cd ..
python -m src.pretraining.pretrain_physionet epochs=1 samples_per_epoch=64 batch_size=32 mask_threshold=0.01 insignificant_prob=0.9
