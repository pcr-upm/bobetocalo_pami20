[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-task-head-pose-estimation-in-the-wild-1/head-pose-estimation-on-aflw)](https://paperswithcode.com/sota/head-pose-estimation-on-aflw?p=multi-task-head-pose-estimation-in-the-wild-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-task-head-pose-estimation-in-the-wild-1/face-alignment-on-aflw2000)](https://paperswithcode.com/sota/face-alignment-on-aflw2000?p=multi-task-head-pose-estimation-in-the-wild-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-task-head-pose-estimation-in-the-wild-1/face-alignment-on-aflw2000-3d)](https://paperswithcode.com/sota/face-alignment-on-aflw2000-3d?p=multi-task-head-pose-estimation-in-the-wild-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-task-head-pose-estimation-in-the-wild-1/face-alignment-on-cofw)](https://paperswithcode.com/sota/face-alignment-on-cofw?p=multi-task-head-pose-estimation-in-the-wild-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-task-head-pose-estimation-in-the-wild-1/pose-estimation-on-300w-full)](https://paperswithcode.com/sota/pose-estimation-on-300w-full?p=multi-task-head-pose-estimation-in-the-wild-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-task-head-pose-estimation-in-the-wild-1/head-pose-estimation-on-biwi)](https://paperswithcode.com/sota/head-pose-estimation-on-biwi?p=multi-task-head-pose-estimation-in-the-wild-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-task-head-pose-estimation-in-the-wild-1/head-pose-estimation-on-aflw2000)](https://paperswithcode.com/sota/head-pose-estimation-on-aflw2000?p=multi-task-head-pose-estimation-in-the-wild-1)

# Multi-task head pose estimation in-the-wild

We provide C++ code in order to replicate the head-pose experiments in our paper https://ieeexplore.ieee.org/document/9303369

If you use this code for your own research, you must reference our PAMI paper:

```
Multi-task head pose estimation in-the-wild
Roberto Valle, José M. Buenaposada, Luis Baumela.
IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI 2020.
https://doi.org/10.1109/TPAMI.2020.3046323
```

#### Requisites
- faces_framework https://github.com/bobetocalo/faces_framework
- ert_simple https://github.com/bobetocalo/ert_simple
- Tensorflow (v1.8.0)

#### Installation
This repository must be located inside the following directory:
```
faces_framework
    └── alignment
        └── ert_simple
    └── multitask 
        └── bobetocalo_pami20
```
You need to have a C++ compiler (supporting C++11):
```
> mkdir release
> cd release
> cmake ..
> make -j$(nproc)
> cd ..
```
#### Usage
Use the --measure option to set the face alignment normalization.

Use the --database option to load the proper trained model.
```
> ./release/face_multitask_bobetocalo_pami20_test --measure pupils --database cofw
```
