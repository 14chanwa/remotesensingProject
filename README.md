# remotesensingProject


This repository implements some depth map estimation algorithm using 3D light fields, originally proposed by Kim et al. in [Scene Reconstruction from High Spatio-Angular Resolution Light Fields](https://www.disneyresearch.com/publication/scene-reconstruction-from-high-spatio-angular-resolution-light-fields/) (2013).


<p align="center">
<img src="https://raw.githubusercontent.com/14chanwa/remotesensingProject/master/report/images/SkysatLR18_240_img/1521805051081_dmap_050.png" width="800">
</p>
<p align="center"><em>Some sample satellite image and the computed disparity map</em></p>


## Dependancies


* OpenCV 3.x should be installed and findable.

* The program should be compiled with C++11 standards. In particular, this program makes use of `<experimental/filesystem>` and its corresponding library `stdc++fs`.

* OpenMP.


## Minimal working example (with the right paths)


Assume we are in the folder containing `README.md`. The following commands builds the library and the tests and run the library's "Hello World!".

```
mkdir build
cd build
cmake ../RSLightFields
make
./test_read_tiff 0
./test_read_tiff 1
```

