# Bunched LPCNet (implementation with Pytorch)

This repository provides ***UNOFFICIAL Bunched LPCNet*** implementations with ***Pytorch***.

- Source of the paper: [Bunched LPCNet: Vocoder for Low-cost Neural Text-To-Speech Systems](https://arxiv.org/abs/2008.04574)

## Introduction

1. For BunchedLPCNet, we set (S=2, B=(8, 0)). It can achieve about **1.5X** faster than the original LPCNet.
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
    - When it is from ***[Tacotron](https://github.com/BridgetteSong/ExpressiveTacotron)***:
        - `python test_vocoder feature.npy`
    - When it is from a wav file:
        - `import numpy as np`
        - `import soundfile as sf`
        - `sox a.wav -b 16 -c 1 -r 16k -t raw -> test_input.s16`
        - `./dump_data -test test_input.s16 test_features.f32`
        - `./lpcnet_demo -synthesis test_features.f32 test_features.pcm`
        - `a = np.fromfile('out.pcm', dtype=np.int16)`
        - `sf.write("out.wav", a, 16000, "PCM_16")`
   
## Reference

1. [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://jmvalin.ca/papers/lpcnet_icassp2019.pdf)
2. [Bunched LPCNet: Vocoder for Low-cost Neural Text-To-Speech Systems](https://arxiv.org/abs/2008.04574)
3. <https://github.com/mozilla/LPCNet>
4. <https://github.com/shakingWaves/LPCNet_torch>
