# Adversarial Machine Learning With Tensorflow
In this repo you can find a custom tensorlfow implementation of various *adversarial attacks* and *adversarial training*.
The repo consists of a python package `adverarial_ml` and a jupyter notebook demo in `demo.ipynb`.
The python package has two python modules: (1) `adverarial_attacks.py`which implements the adversarial attacks (2)
`custom_model.py` which implements a subclass of `tf.keras.Model` featuring an adversarial training option. 

## Who Is This For
This repository may be useful or intersting for you if
- you are looking for an implementation of adversarial attacks or the adversarial
training algorithm (see `adversarial_ml` package)
- interested in seeing adversarial attack and defense 
results on MNIST (see `demo.ipynb`)

## Getting Started
First create your directory for the project and navigate into the directory.

```commandline
mkdir project
cd project
```

Then you can clone the repository into the directory.

```commandline
git clone https://github.com/skmda37/Adversarial_Machine_Learning_Tensorflow.git
```
Finally create a virtual environment `env` with pip and install the requirements
in `requirements.txt` with pip.

```commandline
pip venv env                
source env/bin/activate
pip install -r requirements.txt
```
## Demo
The jupyter notbook `demo.ipynb` serves both as a tutorial on how to use the `adverarial_ml` package and adversarial
machine learning in general. The demo evaluates different models (fully connected neural networks, convolution
neural networks) with and without adversarial training on the adversarial attacks listed below. The dataset used in the
experiments is `MNIST`. After the demo you should be able to see which adversarial attacks are the hardest to defend
against, which type of adversarial examples is most effective for adversarial training and how the adversarial attacks
differ in computational cost. Even though the demo works with the MNIST dataset,
you should be able to use the `adversarial_ml` package for images with arbitrary channel dimension.

## List of attacks that were implemented

- Fast Gradient Sign Method ([Explaining And Harnessing Adversarial Examples - Goodfellow et al](https://arxiv.org/pdf/1412.6572.pdf))
- Step Least Likely ([Adversarial Machine Learning at Scale - Kurakin et al.](https://arxiv.org/pdf/1611.01236.pdf))
- Basic Iterative Method ([Adversarial Machine Learning at Scale - Kurakin et al.](https://arxiv.org/pdf/1611.01236.pdf))
- Iterative Least Likely ([Adversarial Machine Learning at Scale - Kurakin et al.](https://arxiv.org/pdf/1611.01236.pdf))
- Random Plus FGSM ([Fast Is Better Than Free: Revisiting Adversarial Training - Rice and Wong et al.](https://arxiv.org/pdf/2001.03994.pdf))
- PGD With Random Restarts ([https://arxiv.org/pdf/1706.06083.pdf](https://arxiv.org/pdf/1706.06083.pdf))

## Usage 

Each adversarial attack is implemented as a class in `adversarial_attacks.py`. 
The module `custom_model.py` defines the class `CustomModel` which is a subclass to `tf.keras,Model`. 

Let's import the modules and demonstrate how to adversarially train a model (using the Random Plus FGSM attack as an example) and 
evalaute the adversarial robustness.

```python
from adversarial_ml import adversarial_attacks.py as attacks
from adversarial_ml import custom_model as models
```

Let's load the MNIST dataset for the demonstration.
```python
(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess
x_train = tf.constant(x_train.reshape(60000,28, 28,1).astype("float32") / 255)
x_test = tf.constant(x_test.reshape(10000, 28, 28, 1).astype("float32") / 255)

y_train = tf.constant(y_train.astype("float32"))
y_test = tf.constant(y_test.astype("float32"))
```
Next let's define a convolutional neural network that we can adversarially train on MNIST.

```python
# Get adversarial training parameters
eps = 0.3
attack_kwargs = {"eps": eps, "alpha":1.25*eps}
adv_training_with = {"attack": attacks.RandomPlusFgsm,
                     "attack kwargs": attack_kwargs,
                     "num adv": 16}

# Define forward pass
inputs = tf.keras.Input(shape=[28,28,1]
                            dtype=tf.float32, name="image")
x = inputs
x = tf.keras.layers.GaussianNoise(stddev=0.2)(x)

# Convolutional layer followed by 
for i, num_filters in enumerate([32,64,64):
x = tf.keras.layers.Conv2D(
    num_filters, (3,3), activation='relu')(x)
if i < len([32,64,64) - 1:
    # max pooling between convolutional layers
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

x = tf.keras.layers.Flatten()(x)

for num_units in [64]:
   x = tf.keras.layers.Dense(num_units, activation='relu')(x)
   
pred = tf.keras.layers.Dense(10, activation='softmax')(x)

# Get model
my_model = models.CustomModel(inputs=inputs, outputs=pred, 
                              adv_training_with=adv_training_with)
```

Now we want to train `my_model` adversarially. We just do thhis as we would do with any `tf.keras.Model` and it will train using
adversarial training with all the hyperparmeters passed in `adv_training_with`.

```python
# Training parameters
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
METRICS = [tf.keras.metrics.SparseCategoricalAccuracy]
OPTIMIZER = tf.keras.optimizers.RMSprop()

# Compile model
my_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=["accuracy"])

# Fit model to training data 
my_model.fit(x_train, y_train, batch_size=32, epochs=4, valiadation_split=0.2)

# Evaluate model on test data
print("\n")
evaluation = my_model.evaluate(x_test,y_test, verbose=2)
```
![][]

Lastly we would like to perform an adversarial robustness test. For fun's sake we can
also visualize how `my_model` performs on 20 adversarial examples for the most powerful attack *PGD With Random Restarts*. I implemented a method
`attacks.attacks_visual_demo` which does that. This test is informal and just a visulaization to see what's going on.

```python
# Attack to be tested
Attack = attacks.PgdRandomRestart
# Attack parameters
attack_kwargs = {"eps": 0.25, "alpha": 0.25/40, "num_iter": 40, "restarts": 10}

attacks.attack_visual_demo(my_model, Attack, attack_kwargs,
                           x_test[:20], y_test[:20])
```
After running this code in a jupyter notebook cell yo would get this plot:

![attack visualization]["images/attack_visulization.png"]

Lastly let us perform a rigorous adversarial robustness test. This is easy since every instance of `models.CustomModel` has 
a built in method `test_adv_robustness` which prints accuracy results on adversarial attacks with test data for each attack implemented
in `adversarial_attacks.py`. If your computational resources are limited you may want to test ona smaller number of test data like `x_test[:100], y_test[:100]`.
The iterative methods are computationally expensive in particular the *PGD with random restarts* attack.

```pyhton
my_model.test_adv_robustness(x_test, y_test, eps=0.3)
```

The result could look like this:

![][]
