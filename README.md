16322-3d-pose
=============

16.322 3D Pose Estimation Project - A 3D attitude estimator based on the Unscented Kalman Filter (UKF) using a MEMS IMU and magnetometer.

This repo contains a implementation of [Edgar Kraft's quaternion-based UKF](http://kodlab.seas.upenn.edu/uploads/Arun/UKFpaper.pdf) and a program for running the UKF on simulated data or saved experimental data. It also contains sensor drivers for the MPU-9150 on the BeagleBone Black, quaternion utilities, and tools for sensor characterization.


Installation
------------

Clone the repo, and its submodules (you need to include the `--recursive` flag to get the submodules):

```shell
git clone --recursive git@github.mit.edu:mvernacc/16322-3d-pose.git
```

Install pip (if you don't already have it):

```shell
sudo apt-get install python-pip
```

Install 16322-3d-pose using [pip](https://pip.pypa.io/en/stable/). This will automatically install the dependencies.

```shell
cd 16322-3d-pose
pip install -e .
```


Usage
-----

### Estimator
The estimator program is in the top-level directory:

```shell
cd 16322-3d-pose
```

To run the estimator on saved experimental sensor data:

```shell
ipython run_estimator.py -- --meas_source pickle --pkl_file data/exemplars/xyz90.p --use_mag --mag_cal data/exemplars/mag_cal.p
```
To run the estimator on simulated data:

```shell
ipython run_estimator.py -- --meas_source sim --sim_type xyz90 --use_mag
```

To see all options for running the estimator:

```shell
ipython run_estimator.py -- -h
```

### Sensor Characterization
The sensor characterization programs are in the `estimators/sensor_models` directory:

```shell
cd 16322-3d-pose/estimators/sensor_models
```

To run Allan variance analysis on the rate gyroscope:

```shell
ipython allan.py -- ../../data/exemplars/data_for_allan.p gyro_data
```

To run magnetometer calibration:
```shell
ipython mag_cal.py -- --meas_source pickle --pkl_file ../../data/exemplars/data_for_mag_cal.p --solver leastsq
```


Dependencies
------------
 * python 2.7

 * numpy

 * scipy

 * matplotlib

 * [beaglebone-mpu9x50](https://github.com/mvernacc/beaglebone-mpu9x50) for sensor drivers.

 * [beaglebone-px4flow](https://github.mit.edu/mvernacc/beaglebone-px4flow-i2c) for sensor drivers.

 * [estimators](https://github.mit.edu/mvernacc/estimators) for Kalman Filter ans sensor models.

 * [transforms3d](https://github.com/matthew-brett/transforms3d) for quaternion math.
