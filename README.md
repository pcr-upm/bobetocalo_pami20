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

#### Installation
This repository must be located inside the following directory:
```
faces_framework
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
Use the --database option to load the proper trained model.
```
> ./release/face_multitask_bobetocalo_pami20_test --database cofw
```
