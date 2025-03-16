#!/bin/bash

python classifier.py --fine-tune-mode "last-linear-layer" --batch_size 64 --lr 1e-3 --epochs 10 --use_gpu | tee -a last_linear_layer_output.txt
python classifier.py --fine-tune-mode "full-model" --batch_size 64 --lr 1e-5 --epochs 10 --use_gpu | tee -a full_model_output.txt
