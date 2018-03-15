# remotesensingProject


This repository implements some depth map estimation algorithm using 3D light fields, originally proposed by Kim et al. in [Scene Reconstruction from High Spatio-Angular Resolution Light Fields](https://www.disneyresearch.com/publication/scene-reconstruction-from-high-spatio-angular-resolution-light-fields/) (2013).


<p align="center">
<img src="https://raw.githubusercontent.com/14chanwa/remotesensingProject/master/report/animate_mansion_resized/1520877118843_dmap_050.png" width="800">
</p>
<p align="center"><em>Some sample image after the first step of the pipeline (from the original dataset of Kim et al.)</em></p>


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

