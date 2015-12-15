16322-3d-pose
=============

16.322 3D Pose Estimation Project


Usage
-----

The estimator interface is in the `mvernacc` directory:

```shell
cd 16322-3d-pose/mvernacc
```

To run the estimator on saved experimental sensor data:

```shell
ipython run_estimator.py -- --meas_source pickle --pkl_file data/mpu_data_2015-12-15_12-15-19_xyz90.p --use_mag --mag_cal data/mag_cal_2015-12-15_outdoor.p
```
To run the estimator on simulated data:

```shell
ipython run_estimator.py -- --meas_source sim --sim_type xyz90 --use_mag
```

To see all options for running the estimator:

```shell
ipython run_estimator.py -- -h
```

Dependencies
------------
python 2.7
numpy
scipy
matplotlib
beaglebone-mpu9x50
beaglebone-px4flow
estimators
transforms3d
TRICAL
    cmake, svn for TRICAL
    in order to build TRICAL I needed to modify the compiler flags in
    `set(CMAKE_C_FLAGS, "...")` in `CMakeLists.txt`:
     * add `-std=C99`
     * remove `-Weverything`
     * remove `-Wno-documentation`
