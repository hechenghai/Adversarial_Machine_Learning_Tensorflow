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

## Usage 
Each adversarial attack is implemented as a class in `adversarial_attacks.py`.
To illustrate how a specific adversarial attack is  used we take
*Random Plus FGSM* attack as an examples.
Let's import the module `adaversarial_attacks.py` to use the attack:

```python
from adversarial_ml import adversarial_attacks.py as attacks
```

You initialize the attack by passing the `model` which is attacked and the hyperparameters 
of the attack `eps` (maximum perturbation) and `alpha` (step size):

```python
# eps and alpha are float numbers - model is instance of tf.keras.Model
random_plus_fgsm = attacks.RandomPlusFgsm(model=model, eps=esp, alpha=alpha)
```

You can then generate adversarial examples by calling the attack object. It needs as input `clean_images` (images that will be transformed to adversarial examples) and `true_labels` (the true labels of `clean_images`):

```python
# clean_images and true_labels are tf.Tensor of shape (n,h,w,c) and (n,) respectively
adv_examples = random_plus_fgsm(clean_images=clean_images, true_labels=true_labels) # adv_examples.shape = (n,h,w,c)
```


## The implementation of adversarial training
The module `custom_model.py` defines the class `CustomModel` which is a subclass to `tf.keras,Model`.  
Let's import the module `custom_model.py` and show how it can be used:

```python
from adversarial_ml import custom_model as models
```

Next we will initialize a simple model `model_A` that will be trained *without* adversarial training:
```python
# Define forward call for model
inputs = tf.keras.Input(shape=[28,28,1], dtype=tf.float32, name="image")
x = inputs
x = tf.keras.layers.Dense(60, activation='relu')(x)
outputs = tf.keras.layers.Dense(hparams.num_classes, activation='softmax')(x)

# Get model
model_A = models.CustomModel(inputs= inputs, outputs=outputs, adv_training_with=None)
```

Now say you would like to compare this model to another `model_B` of the same architecture that will be trained
*with* adversarial training using the FGSM  with `eps = 0.3` to generate adversarial examples:

```python
# Define adversarial training parameters
eps = 0.3
attack_kwargs = {"eps": eps}
adv_training_with = {"attack": attacks.Fgsm,         # Type of attack used for adversarial examples in training batch
                     "attack kwargs": attack_kwargs, # Parameter for adversarial attack
                     "num adv": 16}                  # Number of adversarial examples in training batch

# Define forward call for model
inputs = tf.keras.Input(shape=[28,28,1], dtype=tf.float32, name="image")
x = inputs
x = tf.keras.layers.Dense(60, activation='relu')(x)
outputs = tf.keras.layers.Dense(hparams.num_classes, activation='softmax')(x)

# Get model
model_A = models.CustomModel(inputs= inputs, outputs=outputs,
                             adv_training_with=adv_training_with)
```

Finally we want to train both models on training data `(x_train, y_train)` and evaluate the adversarial robustness
of both models on test data `(x_test,y_test)`:

```python
# Define training specifics
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy]
optimizer = tf.keras.optimizers.RMSprop()
EPOCHS = 4

# Compile model
for model in [model_A, model_B]:
    model.compile(optimizer=optimizer,
                  loss=loss, metrics=["accuracy"])
    model.fit(x_train, y_train,
                       batch_size=32,epochs=EPOCHS,
                       validation_split=0.2)
    # Evaluate model
    print("\n")
    evaluation = model.evaluate(x_test,y_test, verbose=2)
    
    # Test adversarial robustness
    print("\n")
    model.test_adv_robustness(x_test, y_test)
```
Notice that the training procedure calls `.compile()`, `.fit()`  and `.evaluate()` like any tf.keras.Model would since 
our model is specified by the Subclass API. To evaluate the adversarial robustness we call `.test_adv_robustness()`
which will print out accuracies on adversarial examples generated from `(x_test,y_test)` for all the adversarial attacks
implemented in the module `adversarial_attacks.py`.
