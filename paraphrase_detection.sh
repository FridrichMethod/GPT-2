#!/bin/bash

# python paraphrase_detection.py --batch_size 64 --model_size gpt2 --lr 1e-5 --epochs 10 --use_gpu | tee -a paraphrase_detection_output.txt
# python paraphrase_detection.py --batch_size 64 --model_size gpt2 --lr 1e-5 --epochs 10 --use_gpu --use_lora | tee -a paraphrase_detection_output.txt
# python paraphrase_detection.py --batch_size 64 --model_size gpt2-medium --lr 1e-5 --epochs 10 --use_gpu | tee -a paraphrase_detection_output.txt
# python paraphrase_detection.py --batch_size 64 --model_size gpt2-medium --lr 1e-5 --epochs 10 --use_gpu --use_lora | tee -a paraphrase_detection_output.txt
# python paraphrase_detection.py --batch_size 32 --model_size gpt2-large --lr 1e-5 --epochs 10 --use_gpu | tee -a paraphrase_detection_output.txt
python paraphrase_detection.py --batch_size 32 --model_size gpt2-large --lr 1e-5 --epochs 10 --use_gpu --use_lora | tee -a paraphrase_detection_output.txt
