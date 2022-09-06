# vae-anomaly-detection-for-timeseries

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vae-anomaly-detection?style=flat-square)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/Michedev/VAE_anomaly_detection/Python%20test?style=flat-square)

[中文文档](https://blog.csdn.net/smileyan9/article/details/126668385)

Tensorflow 2.x for timeseries implementation of Variational AutoEncoder for anomaly detection following the paper 《[Variational Autoencoder based Anomaly Detection using Reconstruction Probability](https://www.semanticscholar.org/paper/Variational-Autoencoder-based-Anomaly-Detection-An-Cho/061146b1d7938d7a8dae70e3531a00fceb3c78e8)》.

## dependencies

* tensorflow 2.x
* numpy
* pandas
* sklearn

## usage

Firstly, clone this repository into your local environment.

```bash
$ git clone git@github.com:smile-yan/vae-anomaly-detection-for-timeseries.git
```

Then make sure the dependencies are installed.

```bash
$ cd vae-anomaly-detection-for-timeseries
$ pip install -r requirement.txt
```

Lastly we can run this project as follows:

```bash
$ python main.py
```

## Custom dataset

Make sure that your dataset is a time series set and then do as in `main.py`.

## Q & Α

Any question please push issues here or comment on my blog. > [smileyan9](https://blog.csdn.net/smileyan9/article/details/109255466)

> Smileyan
> 2022.9.6 14:28