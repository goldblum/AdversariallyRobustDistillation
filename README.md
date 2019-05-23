# Adversarially Robust Distillation (ARD): PyTorch implementation

This repository contains Pytorch code for the ARD method from ["Adversarially Robust Distillation"](insert arxiv link) by Micah Goldblum, Liam Fowl, Soheil Feizi, and Tom Goldstein.

Adversarially Robust Distillation is a method for transferring robustness from a robust teacher network to the student network during distillation.  In our experiments, small ARD student models outperform adversarially trained models with identical architecture.

## Prerequisites
* Python3
* Pytorch
* CUDA

## Run
Here is an example of how to run our program:
```
$ python main.py --teacher_path INSERT-YOUR-TEACHER-PATH
```
## Want to attack ARD?
A MobileNetV2 ARD model distilled from a [TRADES](https://arxiv.org/pdf/1901.08573.pdf) WideResNet (34-10) teacher on CIFAR-10 can be found [here](https://drive.google.com/drive/folders/15Od-zi6HGwQoIym3AkLGzLVPaR8oH9UR?usp=sharing).
