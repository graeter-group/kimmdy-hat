# HAT reaction plugin

## Introduction

Plugin for KIMMDY to perform hydrogen atom transfer reactions. Loads a trained tensorflow model to predict energy barriers.

## Installation

* Download
* Presetup:

    ```bash
    pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt-libs
    pip install tensorflow[and-cuda]==2.15.0
    ```

* `pip install -e ./`  or for development: `pip install -r requirements.txt`
* Lint with black



## Requirements

The installation and execution of the HAT plugin depends on the TensorFlow and CUDA version available in your environment. To ensure compatibility, make sure to load modules before installation in the same way as you will during runtime.

Models were built and saved in keras 2 and would need to be migrated for use in keras 3. Therefore use Tensorflow <2.16 and appropriate kgcnn versions.



