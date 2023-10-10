# DCGAN for MNIST Dataset and Custom Image Datasets

This repository contains an implementation of a DCGAN (Deep Convolutional Generative Adversarial Network) trained on the MNIST dataset and designed to work with custom image datasets as well. The DCGAN is a popular architecture for generating realistic images by training a generator and discriminator network in an adversarial manner.

![68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313030302f312a33394e6e6e695f6e685044614c7539416e544c6f57772e706e67](https://github.com/mohammadr8za/DC-GAN/assets/72736177/6ac5090f-b12c-45b5-af36-21d7f1b3f298)

## Key Features:

* MNIST Dataset and Custom Image Datasets: While the DCGAN's results on the MNIST dataset are showcased for validation purposes, this implementation provides the flexibility to train the model on any other custom image dataset. Simply provide the address of the custom dataset in the code, and the DCGAN will adapt accordingly.

* Generator and Discriminator Networks: The DCGAN architecture is implemented using two convolutional networks - a generator and a discriminator. The generator takes random noise as input and generates synthetic images, while the discriminator aims to distinguish between real and generated images.

* Deep Convolutional Networks: Both the generator and discriminator networks utilize deep convolutional layers to capture complex image features. This allows the model to learn hierarchical representations and generate high-quality images.

* Adam Optimizer: The Adam optimizer is employed for training the DCGAN. It is a popular choice due to its adaptive learning rate and momentum properties, which help to stabilize and speed up the training process.

## Usage:

* Clone the repository and install the required dependencies.

* Customize the code to provide the address and specific requirements of your custom image dataset for training the DCGAN.

* Run the "main.py" script, which will train the generator and discriminator networks in an adversarial manner using either the MNIST dataset or your own custom dataset.

* Once training is complete, you can generate new images using the trained generator network.

## Results:

By training the DCGAN on the MNIST dataset, the generator network learns to generate realistic-looking handwritten digits. This serves as a validation of the DCGAN's performance. However, the code's flexibility allows you to train the DCGAN on any custom image dataset, making it adaptable to a wide range of applications.

## Future Improvements:

This repository provides a foundation for implementing and training DCGANs on both the MNIST dataset and custom image datasets. Future improvements could include exploring different architectural variations, incorporating advanced techniques such as conditional generation or spectral normalization, or optimizing the code for improved performance and training stability.

Feel free to further modify and expand upon this description to accurately reflect your implementation and any additional details you want to include.
