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
Each adversarial attack is implemented as subclass of the class `AdversarialAttack`. Below is a snippet that shows the class implementation

```Python
class AdversarialAttack:
    def __init__(self, model, eps):
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()  # Loss that is used for adversarial attack
        self.model = model      # Model that is used for generatig the adversarial examples
        self.eps = eps          # Threat radius of adversarial attack
        self.specifics = None   # String that contains all hyperparameters of attack
        self.name = None        # Name of the attack - e.g. FGSM
```
To illustrate how a specific adversarial attack is implemented and used we show the code for the *Random Plus FGSM* attack. 

```Python
class RandomPlusFgsm(AdversarialAttack):
    def __init__(self, model, eps, alpha):
        super().__init__(model, eps)
        self.name = "Random Plus FGSM"
        self.specifics = "Random Plus FGSM - eps: {:.2f} - alpha: {:.4f}".format(eps, alpha)
        self.alpha = alpha

    def __call__(self, clean_images, true_labels):
        # Sample initial perturbation uniformly from interval [-epsilon, epsilon]
        random_delta = 2 * self.eps * tf.random.uniform(shape=clean_images.shape) - self.eps
        # Add random initial perturbation
        X = clean_images + random_delta
        # Compute Gradients
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # Only gradient w.r.t clean_images is accumulated NOT w.r.t model parameters
            tape.watch(X)
            prediction = self.model(X)
            loss = self.loss_obj(true_labels, prediction)
            
        gradients = tape.gradient(loss, X)
        perturbation = self.alpha * tf.sign(gradients)
        # Add perturbation
        X = X + perturbation
        # Make sure we don't leave epsilon L infinity ball around X
        X = tf.clip_by_value(X, clean_images - self.eps, clean_images + self.eps)
        # Make sure entries remain between 0 and 1
        X = tf.clip_by_value(X, 0, 1)
        return X
```


## The implementation of adverarial training
