#!/bin/bash

python paraphrase_detection.py --batch_size 64 --model_size gpt2 --lr 1e-5 --epochs 10 --use_gpu | tee -a paraphrase_detection_output.txt
