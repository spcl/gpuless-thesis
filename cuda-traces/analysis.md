# Analysis

| Application | Kernel launches | Unique Kernels | cudaMalloc | Copy H2D | Copy D2D | Copy D2H | CUDA API time | Total time
| --- | --- | --- | --- | --- | --- | --- |
| hotspot.log | 25 | 1 | 3 | 2 | 0 | 1 | 3.615 ms | 35.202 ms |
| dwt2d.log | 10 | 4 | 12 | 1 | 9 | 3 | 1.417 ms | 47.409 ms |
| srad2.log | 4 | 2 | 6 | 2 | 0 | 2 | 5.609 ms | 52.25 ms |
| resnet50.log | 310 | 21 | 37 | 321 | 3 | 3 | 15.676 ms | 139.005 ms |
| bert-squad.log | 714 | 22 | 74 | 397 | 1 | 3 | 132.906 ms | 531.512 ms |

# Description

## HotSpot

From [Rodinia: HotSpot](http://www.cs.virginia.edu/rodinia/doku.php?id=hotspot).

HotSpot is a widely used tool to estimate processor temperature based on an
architectural floorplan and simulated power measurements. The thermal
simulation iteratively solves a series of differential equations for block.
Each output cell in the computational grid represents the average temperature
value of the corresponding area of the chip. Our CUDA implementation
re-implements the transient thermal differential equation solver from HotSpot.

## Dwt2d

From [Rodinia: GPUDWT](http://www.cs.virginia.edu/rodinia/doku.php?id=gpudwt).

Discrete Wavelet Transform (DWT) is a broadly used digital signal processing
technique with application in diverse areas. GPUDWT library implements forward
and revers DWT 9/7 and 5/3 transforms and supports CUDA platform (version 3.2
and newer) and NVIDIA GPUs. 

##  Srad2

From [Rodinia: SRAD](http://www.cs.virginia.edu/rodinia/doku.php?id=srad).

SRAD (Speckle Reducing Anisotropic Diffusion) is a diffusion method for
ultrasonic and radar imaging applications based on partial differential
equations (PDEs). It is used to remove locally correlated noise, known as
speckles, without destroying important image features.

## Resnet50

Image recognition using Resnet50.

## BERT

BERT for usage with the Stanford Question Answering Dataset (SQUAD).
