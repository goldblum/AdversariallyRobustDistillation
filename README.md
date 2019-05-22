# Adversarially Robust Distillation (ARD): A PyTorch implementation

This repository contains Pytorch code for the ARD method from [paper](Insert Link Once on Arxiv) "Adversarially Robust Distillation" by Micah Goldblum, Liam Fowl, Soheil Feizi, and Tom Goldstein.

Adversarially Robust Distillation is a method for transferring robustness from a robust teacher network to the student network during distillation.  In our experiments, small ARD student models from strong teachers outperform adversarially trained models with identical architecture.

## Prerequisites
* Python3
* Pytorch
* CUDA

## Run
Here is an example of how to run our program:
```
$ python main.py --teacher_path INSERT-YOUR-TEACHER-PATH-HERE
```
