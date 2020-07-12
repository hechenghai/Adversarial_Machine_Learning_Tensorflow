import tensorflow as tf
from adversarial_ml import adversarial_attacks as attacks


class CustomModel(tf.keras.Model):

    def __init__(self, inputs, outputs, adv_training_with=None, **kargs):
        """
        inputs: inputs for buidling tf.keras.Model
        outputs: outputs for building tf.keras.Model
        adv_training_with: None or dictionary with items: ("Attack", Adversarial Attack Class),
        ("attack kwargs", dictionary with all kwargs for call of instance of Adversarial Attack Class
        except for model which is set to self later), ("num adv", number of adversarial examples in
        training batch)
        """

        super(CustomModel, self).__init__(inputs=inputs, outputs=outputs, **kargs)
        # Training information (set to string in __init__)
        self.training_info = None

        # Adversarial training specifics
        self.adv_training_with = adv_training_with

        if self.adv_training_with != None:
            Adv_attacks = [attacks.Fgsm, attacks.OneStepLeastLikely,
                           attacks.RandomPlusFgsm, attacks.BasicIter,
                           attacks.IterativeLeastLikely]
            Attack = self.adv_training_with["attack"]
            assert Attack in Adv_attacks
            attack_kwargs = adv_training_with["attack kwargs"]
            self.generate_adv_examples = Attack(model=self, **attack_kwargs)
            self.training_info = "adversarially trained with " + \
                                 self.generate_adv_examples.specifics
        else:
            self.training_info = "trained without adversarial examles"

    @tf.function
    def train_step(self, data):
        # Unpack images x and labels y
        x, y = data

        if self.adv_training_with != None:
            # Get number of adversarial images for training batch
            k = self.adv_training_with["num adv"]
            # Get adversarial examples
            adv_x = self.generate_adv_examples(x[:k], y[:k])
            # Get clean images
            clean_x = x[k:]
            # Make new traininig batch
            x = tf.concat([adv_x, clean_x], axis=0)

        # Track Gradients w.r.t weights
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)
            # Compute the loss value
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients w.r.t weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        # Return dict mapping metric names to current values
        return {m.name: m.result() for m in self.metrics}

    def test_adv_robustness(self, test_images, test_labels, eps=0.3):
        assert (test_images.shape[0],) == test_labels.shape
        # Get list of adversarial attacks for test
        attack_list = [attacks.Fgsm, attacks.RandomPlusFgsm,
                   attacks.BasicIter,
                   attacks.IterativeLeastLikely,
                   attacks.OneStepLeastLikely]

        # Get attack parameters
        attack_params = [{"model": self, "eps": eps},  # Fgsm kwargs
                         {"model": self, "eps": eps, "alpha": 1.25 * eps},  # Random Plus Fgsm kwargs
                         {"model": self, "eps": eps, "alpha": eps / 40, "num_iter": 40},  # Basic Iter kwargs
                         {"model": self, "eps": eps, "alpha": eps / 40, "num_iter": 40},  # IterativeLeastLikely kwargs
                         {"model": self, "eps": eps}]  # OneStepLeastLikely kwargs

        # Initialize adversarial attacks with parameters
        attack_list = [Attack(**params) for Attack, params in
                   zip(attack_list, attack_params)]

        # Get inputs for attack in attacks
        attack_inputs = 3 * [(test_images, test_labels)] + 2 * [(test_images,)]

        # Get number of test images
        num_images = test_labels.shape[0]

        # Test adversarial robustnes
        print("Test adversarial robustness for model trained" + self.training_info)
        for attack, attack_input in zip(attack_list, attack_inputs):
            adv_examples = attack(*attack_input)
            pred = self(adv_examples)
            pred = tf.math.argmax(pred, axis=1)
            equality = tf.math.equal(pred, tf.cast(test_labels, tf.int64))
            accuracy = tf.math.reduce_sum(tf.cast(equality, tf.float32)).numpy() / num_images
            print(100 * "=")
            print(attack.specifics + f" - accuracy: {accuracy}")


