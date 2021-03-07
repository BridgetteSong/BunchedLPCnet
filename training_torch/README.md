# Bunched LPCNet (implementation with Pytorch)

This repository provides ***UNOFFICIAL Bunched LPCnet*** implementations with ***Pytorch***.

- Source of the paper: <https://arxiv.org/abs/2008.04574>

## Introduction

1. For BunchedLPCNet, we set (S=2, B=(8, 0)). It can achieve 1.5X than the original LPCNet.
2. It is based on the open source [LPCNet](https://github.com/mozilla/LPCNet/commit/bffdcee95b4303167a34007ea22c8d304ca204da).
3. not support '***Encoder Mode***' and '***Decoder Mode***'

## Training and Test

1. prepare data according the original LPCNet and set data_path in **hparams.py**
2. training model
    - `cd training_torch`
    - `python train_lpcnet.py`
3. dump model
    - `python dump_lpcnet -c checkpoint`
4. rebuild
    - `cd ../`
    - `make`
5. test
    - prepare your feature file, confirm your "**feature.shape[-1] = 20**"
    - When it is from ***Tacotron***:
        - `python test_vocoder feature.npy`
    - When it is from a wav file:
        - `sox a.wav -b 16 -c 1 -r 16k -r -raw -> test_input.s16`
        - `./dump_data -test test_input.s16 test_features.f32`
        - `feature = np.fromfile(test_features.f32, dtype=np.float32).reshape(-1, 55)`
        - `feature = np.concatenate((feature[:,:18], feature[:, 18:20]), axis=-1)`
        - `np.save('feature.npy', feature)`
        - `python test_vocoder feature.npy`
   
## Reference

1. [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://jmvalin.ca/papers/lpcnet_icassp2019.pdf)
2. [Bunched LPCNet: Vocoder for Low-cost Neural Text-To-Speech Systems](https://arxiv.org/abs/2008.04574)
3. <https://github.com/mozilla/LPCNet>
4. <https://github.com/shakingWaves/LPCNet_torch>