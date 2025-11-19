---

# ٤ MNIST-MapNet: Handwritten Digit Recognition in MATLAB

This repository implements a custom 4-layer “mapnet” neural network in MATLAB to classify handwritten digits from the MNIST dataset.
The network uses a ReLog nonlinearity, custom backpropagation, and the Adam optimizer, and achieves under **1.6%** test error in the best runs.

This project was completed for **PSL432 — Theoretical Physiology** at the University of Toronto (2025).

---

## ✅ Features

* 4-layer, convolution-style feedforward network:

  * Input: 784 units (28×28 pixels)
  * Hidden layer 1: 784 units
  * Hidden layer 2: 784 units
  * Output: 10 units (digits 0–9)
* Custom ReLog activation function and gradient
* Sparse, receptive field–based connectivity in hidden layers
* Output layer connected to both hidden layers 2 and 3
* Adam optimization with bias correction
* Fully vectorized minibatch training on MNIST
* Tracks training and test errors across epochs

---

## 👀 File Overview

* `zhang_create_mapnet.m`
  Initializes the `net` struct:

  * Layer sizes
  * ReLog activation setting
  * Weights and biases with Kaiming-style initialization
  * Receptive field–based connectivity for hidden layers
  * Full connectivity for the output layer
  * Adam optimizer state variables

* `zhang_forward_relog.m`
  Forward pass:

  * Stores input in `net.activations{1}`
  * Applies weighted connections + ReLog activation in hidden layers
  * Combines outputs from layers 2 and 3 into the final output layer
  * Returns pre-activations (`net.v`) and activations (`net.activations`)

* `zhang_backprop_relog.m`
  Backward pass:

  * Starts from the output error (SoftMax + cross-entropy–style gradient)
  * Propagates deltas backward through layers 3 and 2, including:

    * Special handling for the shared connections from layer 2 and 3 into layer 4
  * Computes `net.dL_dW{l}` and `net.dL_db{l}` for each trainable layer
  * Stores an input-layer delta for debugging

* `adam.m`
  Adam optimizer:

  * Updates first and second moments for weights (`W_`, `W__`) and biases (`b_`, `b__`)
  * Applies bias correction (`adam_1t`, `adam_2t`)
  * Performs parameter updates using the corrected moments

* `MNIST.mat`
  Expected to contain the MNIST training and test data in the format used by the course, e.g.:

  * `TRAIN_images`, `TRAIN_labels`, `TRAIN_answers`
  * `TEST_images`, `TEST_labels`, `TEST_answers`
    (You may need to adjust variable names if your file is different.)

* Main training script (the one you run in MATLAB)
  Contains:

  * Loading `MNIST.mat`
  * Creating the network with `zhang_create_mapnet`
  * Training loop over epochs and mini-batches
  * Calls to `zhang_forward_relog`, `zhang_backprop_relog`, and `adam`
  * Evaluation on the test set and logging `test_errors_per_epoch`

Rename this file to something like `train_mapnet_MNIST.m` or `main.m` if you want a clearer entry point.

---

## 📋 Requirements

* MATLAB (tested with versions compatible with basic matrix operations and `randperm`)
* MNIST dataset saved in `MNIST.mat` with the expected variables:

  * `TRAIN_images`: N_train × 784
  * `TRAIN_labels`: N_train × 10 (one-hot)
  * `TEST_images`: N_test × 784
  * `TEST_labels`: N_test × 10 (one-hot)

Place `MNIST.mat` in the root of the repository or update the `load 'MNIST.mat';` line in the main script.

---

## ❓ How to Run

1. Clone or download this repository.

2. Open MATLAB and add the repository folder to your path, e.g.:

   ```matlab
   addpath(genpath('path/to/this/repo'));
   ```

3. Make sure `MNIST.mat` is in the same directory as your main script (or adjust the path in the `load` command).

4. Open the main training script (e.g. `train_mapnet_MNIST.m`) and run it:

   ```matlab
   run('train_mapnet_MNIST.m');
   ```

5. During training:

   * The script will iterate over 10 epochs
   * Use 600 mini-batches per epoch (batch size 100)
   * Update network parameters with Adam (`eta = 0.001`, `β1 = 0.9`, `β2 = 0.999`)

6. After each epoch, the script evaluates the network on the test set and records:

   * `test_errors_per_epoch(epoch)` = number of misclassified digits

At the end, you should see the test errors per epoch printed in the MATLAB console.

---

## 😊 Results

With the provided architecture and training setup, the network reaches a test error below **1.6%** on MNIST in the best runs (i.e., over 98.4% accuracy).

Your exact results may vary slightly due to random initialization and shuffling.

---

## 👩🏻‍💻 Implementation Details

* **ReLog activation**
  The hidden layers use a ReLog nonlinearity:
  [
  f(x) = \text{sign}(x) \cdot \log(1 + |x|)
  ]
  implemented in a sign-preserving way to keep sensitivity for both positive and negative inputs.

* **Map-style connectivity**
  Hidden layers 2 and 3 use receptive-field based connectivity to mimic convolution-like local processing:

  * `net.connectivity{l}` masks the weight matrices so each neuron only connects to a local patch of the previous layer.
  * The output layer receives concatenated activations from layer 2 and layer 3.

* **Optimization**
  Adam is implemented from scratch:

  * Per-layer first (`W_`, `b_`) and second (`W__`, `b__`) moments
  * Bias-corrected updates using `adam_1t`, `adam_2t`

---

## 💜 Acknowledgements

This project was completed as part of **PSL432 — Theoretical Physiology** at the University of Toronto.
The MNIST dataset is originally from Yann LeCun et al.

---
