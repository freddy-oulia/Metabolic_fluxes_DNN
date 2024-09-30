import tensorflow as tf
from numpy.random import seed


def create_dnn(number_neurons, loss_fn, l_metrics, seed_val=42):
    """
    Create a Deep Neural Network
    :param number_neurons: List with the number of neurons for each layer
    :param loss_fn: Loss function for the model
    :param l_metrics: List of metrics for the model
    :param seed_val: A seed to create the neural network
    :return: A model
    """
    # Fix seed
    seed(seed_val)
    tf.random.set_seed(seed_val)
    kernel = tf.keras.initializers.GlorotUniform(seed_val)  # Initializer for weights matrix

    # Create the neural network
    layer = tf.keras.layers.Input(shape=(3,))
    input_layer = layer
    for i in range(len(number_neurons)):
        layer = tf.keras.layers.Dense(number_neurons[i], kernel_initializer=kernel, activation="elu")(layer)
    output = tf.keras.layers.Dense(1, kernel_initializer=kernel, activation="sigmoid")(layer)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output, name="Glycolysis_model")

    # model.compile(loss=loss_fn, optimizer=tf.keras.optimizers.Adam(), metrics=l_metrics)
    model.compile(loss=loss_fn, optimizer=tf.keras.optimizers.legacy.Adam(), metrics=l_metrics)
    return model
