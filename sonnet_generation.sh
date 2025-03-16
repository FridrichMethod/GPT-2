#!/bin/bash

python sonnet_generation.py --batch_size 64 --model_size gpt2 --lr 1e-5 --epochs 10 --beam_search --use_lora --use_gpu | tee -a sonnet_generation_output.txt
