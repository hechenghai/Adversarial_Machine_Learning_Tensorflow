import tensorflow as tf

class AdversarialAttack:
    def __init__(self, model, eps):
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()  # Loss that is used for adversarial attack
        self.model = model
        self.eps = eps  # Threat radius of adversarial attack
        self.specifics = None
        self.name = None


class Fgsm(AdversarialAttack):
    def __init__(self, model, eps):
        super().__init__(model, eps)
        self.name = "FGSM"
        self.specifics = "FGSM - eps: {:.2f}".format(eps)

    def __call__(self, clean_images, true_labels):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # Only gradien w.r.t clean_images is accumulated NOT w.r.t model parameters
            tape.watch(clean_images)
            prediction = self.model(clean_images)
            loss = self.loss_obj(true_labels, prediction)

        gradients = tape.gradient(loss, clean_images)
        perturbations = self.eps * tf.sign(gradients)

        adv_examples = clean_images + perturbations
        adv_examples = tf.clip_by_value(adv_examples, 0, 1)
        return adv_examples


class OneStepLeastLikely(AdversarialAttack):
    def __init__(self, model, eps):
        super().__init__(model, eps)
        self.name = "One Step Least Likely (Step 1.1)"
        self.specifics = "One Step Least Likely (Step 1.1) - eps: {:.2f}".format(eps)

    def __call__(self, clean_images):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # only gradient w.r.t. clean_images is accumulated NOT w.r.t model parameters!
            tape.watch(clean_images)
            prediction = self.model(clean_images)
            y_ll = tf.math.argmin(prediction, 1)
            loss = self.loss_obj(y_ll, prediction)

        gradients = tape.gradient(loss, clean_images)
        perturbations = self.eps * tf.sign(gradients)

        # assert eps.shape[0] == perturbation.shape[0]
        adv_examples = clean_images - perturbations
        adv_examples = tf.clip_by_value(adv_examples, 0, 1)
        return adv_examples


class BasicIter(AdversarialAttack):
    def __init__(self, model, eps, alpha, num_iter):
        super().__init__(model, eps)
        self.alpha = alpha
        self.num_iter = num_iter
        self.name = "Basic Iterative Method"
        self.specifics = "Basic Iterative Method " \
                         "- eps: {:.2f} - alpha: {:.4f} " \
                         "- num_iter: {:d}".format(eps, alpha, num_iter)

    def __call__(self, clean_images, true_labels):
        X = clean_images
        for i in tf.range(self.num_iter):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                # Only gradients w.r.t. X are accumulated, NOT model parameters
                tape.watch(X)
                prediction = self.model(X)
                loss = self.loss_obj(true_labels, prediction)

            gradients = tape.gradient(loss, X)
            perturbations = self.alpha * tf.sign(gradients)

            X = X + perturbations
            X = tf.clip_by_value(X, clean_images - self.eps, clean_images + self.eps)
            X = tf.clip_by_value(X, 0, 1)
        return X


class IterativeLeastLikely(AdversarialAttack):
    def __init__(self, model, eps, alpha, num_iter):
        super().__init__(model, eps)
        self.alpha = alpha
        self.num_iter = num_iter
        self.name = "Iterative Least Likely (Iter 1.1)"
        self.specifics = "Iterative Least Likely (Iter 1.1) " \
                         "- eps: {:.2f} - alpha: {:.4f} " \
                         "- num_iter: {:d}".format(eps, alpha, num_iter)

    def __call__(self, clean_images):
        X = clean_images
        for i in tf.range(self.num_iter):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                # Only gradients w.r.t. X are accumulated, NOT model parameters
                tape.watch(X)
                prediction = self.model(X)
                y_ll = tf.math.argmin(prediction, 1)
                loss = self.loss_obj(y_ll, prediction)

            gradients = tape.gradient(loss, X)
            perturbations = self.alpha * tf.sign(gradients)

            X = X - perturbations
            X = tf.clip_by_value(X, clean_images - self.eps, clean_images + self.eps)
            X = tf.clip_by_value(X, 0, 1)
        return X


class RandomPlusFgsm(AdversarialAttack):
    def __init__(self, model, eps, alpha):
        super().__init__(model, eps)
        self.name = "Random Plus FGSM"
        self.specifics = "Random Plus FGSM - eps: {:.2f} - alpha: {:.4f}".format(eps, alpha)
        self.alpha = alpha

    def __call__(self, clean_images, true_labels):
        random_delta = 2 * self.eps * tf.random.uniform(shape=clean_images.shape) - self.eps
        X = clean_images + random_delta
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # Only gradien w.r.t clean_images is accumulated NOT w.r.t model parameters
            tape.watch(X)
            prediction = self.model(X)
            loss = self.loss_obj(true_labels, prediction)

        gradients = tape.gradient(loss, X)
        perturbations = self.alpha * tf.sign(gradients)

        X = X + perturbations
        X = tf.clip_by_value(X, clean_images - self.eps, clean_images + self.eps)
        X = tf.clip_by_value(X, 0, 1)
        return X