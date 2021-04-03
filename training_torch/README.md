# Bunched LPCNet (implementation with Pytorch)

This repository provides ***UNOFFICIAL Bunched LPCNet*** implementations with ***Pytorch***.

- Source of the paper: [Bunched LPCNet: Vocoder for Low-cost Neural Text-To-Speech Systems](https://arxiv.org/abs/2008.04574)

## Introduction

1. For BunchedLPCNet, we set (S=2, B=(8, 0)). It can achieve about **1.5X** faster than the original LPCNet.
2. It is based on the open source [LPCNet](https://github.com/mozilla/LPCNet/commit/bffdcee95b4303167a34007ea22c8d304ca204da).
3. Not support '***Encoder Mode***' and '***Decoder Mode***'

## TODO

- [x] Preparing python padding data in the **data_uitls.py** may be not right, but final results is not affected.
- [x] Using higher ulaw(10) and splitting coarse(5) and fine(5) bits result in bad quality. Maybe there are some problems and need further experiments.
- [ ] Some **click sounds** maybe occur occasionally in the generated wav.
- [ ] Using continuous distribution([Gaussian distribution](https://ieeexplore.ieee.org/document/9053337)) need to be modeled instead of discrete softmax distribution.
- [ ] [Multi-band](https://arxiv.org/abs/2005.05551) mode needs to be supported.

## Training and Test

1. prepare data according to the original LPCNet and set data_path in **config.yaml**
2. train model
    - `cd training_torch`
    - `python train_lpcnet.py`
3. dump model
    - `python dump_lpcnet.py -c checkpoint/out_dir/pytorch_lpcnet20_384_10_G16_119.h5`
4. rebuild
    - `cd ../`
    - `make`
5. test
    - prepare your feature file, confirm your "**feature.shape[-1] = 20**"
    - When it is from ***[Tacotron](https://github.com/BridgetteSong/ExpressiveTacotron)***:
        - `python test_vocoder.py feature.npy`
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
