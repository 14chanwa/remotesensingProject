# remotesensingProject


Implements some depth map estimation using 3D light fields.


### Dependancies


OpenCV 3.x should be installed and findable.


### Compiling with the right paths


Assume we are in the folder containing `README.md`:

```
mkdir build
cd build
cmake ../RSLightFields
make
./test_read_tiff 0
./test_read_tiff 1
```

