# Adversarial Machine Learning With Tensorflow
In this repo you will find a custom tensorlfow implementation of various *adversarial attacks* and *adversarial training*. The repos consists of a pyhton package `adverarial_ml` and a demo jupyter notebook `demo.ipynb`. The python packages has two python modules: (1) `adverarial_attacks.py`which implements the adversarial attacks (2) `custom_model.py` which implements a subclass of `tf.keras.Model` which features an adversarial training option. 

## Demo
The jupyter notbook `demo.ipynb` serves both as a tutorial on how to use the `adverarial_ml` package and adversarial machine learning in general. The demo evaluates different models (fully connected neural networks, convolution neural networks) with and without adversarial training on the adverarial attacks listed above.

## List of attacks that were implemented

- Fast Gradient Sign Method ([Explaining And Harnessing Adverarial Examples - Goodfellow et al](https://arxiv.org/pdf/1412.6572.pdf))
- Step Least Likely ([Adverarial Machien Leanring at Scale - Kurakin et al.](https://arxiv.org/pdf/1611.01236.pdf))
- Basic Iterative Method ([Adverarial Machien Leanring at Scale - Kurakin et al.](https://arxiv.org/pdf/1611.01236.pdf))
- Iterative Least Likely ([Adverarial Machien Leanring at Scale - Kurakin et al.](https://arxiv.org/pdf/1611.01236.pdf))
- Random Plus FGSM ([Fast Is Better Than Free: Revisiting Adversarial Training - Rice and Wong et al.](https://arxiv.org/pdf/2001.03994.pdf))

## The implementation of the adversarial attacks


## The implementation of adverarial training
