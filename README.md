# Training and Inference for End-to-End Self-Driving Car Deep Learning Models

## Overview

Self driving cars need to make steering and other control decisions at deterministic pace to be able to safely control the behavior of the vehicle in traffic. A car traveling 70 mph moves one feet every 10 ms. One feet can be the difference between successful corrective action and a fatal accident. In this project we analyze the inference execution time determinism to compare Python and C++ based inference for self driving cars. We expect some fluctuations and fluctuations of the order of 10 ms are probably acceptable but 50 ms fluctuations probably would be too much for such a critical control system.  

In this project we have created a TensorFlow implementation of the [NVIDIA End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) self driving car model. Our implementation is based on the Python TensorFlow implementation by [Sully Chen](https://github.com/SullyChen/Autopilot-TensorFlow).

The project consists of a python based training script and both python and C++ based inference implementations. At the end of the training the training script saves a graph definition where all weights are replaced with corresponding constant values. This saved graph is then loaded by both Python and C++ inference implementations.

## Setting Up

### Checkout TensorFlow Sources

To use the C++ inference, you will need to have TensorFlow source code and all dependencies to be able to build TensorFlow from sources.

```bash
~$ git clone https://github.com/tensorflow/tensorflow
~$ cd tensorflow
~/tensorflow$
```

You will need r1.0 or newer release of TensorFlow.

### Build TensorFlow

Follow the instruction in [TensorFlow Documentation](https://www.tensorflow.org/install/install_sources) on how to download and build TensorFlow from the sources.


**Note:** You don't have to install the TensorFlow pip package build from sources if you already have TensorFlow r1.0 or later installed that you'd prefer to use. However validating the build is difficult without installing the package so at the minimum test the pip package you build in a virtualenv. If the version of TensorFlow you have installed is different from the version of source code you checked out, you may run into problems.

### Checkout This Source Code

After you have TensorFlow build from sources installed, check out this source code under `tensorflow` subfolder of the TensorFlow sources.

```bash
~/tensorflow$ cd tensorflow
~/tensorflow/tensorflow$ git clone https://github.com/tmaila/autopilot.git
```

### Get Dataset

Download the driving  [dataset](https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view?usp=sharing) and extract into the repository folder. After the extraction you should have a `driving_dataset` subfolder under your repository folder.

### Preparing Dataset

We have separated the data preprocessing to its own separate step to purely focus on running the data through the graph in the inference step.

Use `python preprocess.py` to prepare the dataset for training.  

```bash
python preprocess.py
```

## How To Build

The C++ executable for the project is being build using TensorFlow's bazel build system. Use ```bazel build -c opt ...``` to build the executable.

```bash
~/tensorflow/tensorflow$ cd autopilot
~/tensorflow/tensorflow/autopilot$ bazel build -c opt ...
```
You may want to pass optimization flags to bazel. These enable cuda gpu support if not enable by default and some cpu optimizations.

```bash
$ bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda ...
```

## How To Use

### Training the Model

Once everything is set up the model needs to be trained with the prerecorded driving data.

Use `python train.py` to train the model.

```bash
python train.py
```

Training the model should create a save folder that will contain a `graph.pb` and `frozen_graph.pb` files as well as some checkpoint files. The `graph.pb` contains our model without weight variables and the `frozen_graph.pb` contains the same model but with weight variables converted into constants. We use the latter for inference.

### Running the Model Inference

After the model is trained we can run the inference on the model. We are going to run through all the data in the dataset to evaluate the inference execution time performance. Typically you don't want to use your test and validation sets for inference, however that is not a problem in our case as we are not evaluating the prediction performance of our model but purely focused on timing statistics. If you prefer to use some other driving data set for inference, feel free to do so.

#### Python

Use `python run.py` to run the model inference on the dataset using Python.

```bash
python run.py
```

Running the inference through the data will take a while. We are not logging any progress metrics to the screen as it may impact the timing statistics that we are interested in.

### C++

Use the executable we built above to run the inference on the dataset using C++.

```bash
../../bazel-bin/tensorflow/autopilot/autopilot
```

## Analyzing the Results

Running the inferences using Python and C++ will create run log files under the `results` folder. To analyze the timing statistics and compare the inference implementations we have create a Jupyter notebook.

```bash
jupyter notebook autopilot.ipynb
```

To view the precomputed Jupyter notebook, open the autopilot.html file with your
browser.

You can also create the plots shown in the notebook by running the `analyze.py`
script without Jupyter. This is handy when you want to make changes to the
inference implementation and view the outcome of your changes.

```bash
python analyze.py
```
