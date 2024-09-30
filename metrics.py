import tensorflow as tf


class R2(tf.keras.metrics.Metric):
    
    def __init__(self, **kwargs):
        super(R2, self).__init__(name="r2", **kwargs)
        self.r2_score = self.add_weight(name="r2_res", initializer="zeros")
        self.n_batchs = self.add_weight(name="number_batchs", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if not tf.is_tensor(y_true):
            y_true = tf.convert_to_tensor(y_true)
        if not tf.is_tensor(y_pred):
            y_pred = tf.convert_to_tensor(y_pred.flatten())
        
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mean_y_true = tf.reduce_mean(y_true)
        ss_res = tf.math.reduce_sum(tf.math.square(tf.subtract(y_true, y_pred)))
        ss_tot = tf.math.reduce_sum(tf.math.square(tf.subtract(y_true, mean_y_true)))
        self.n_batchs.assign_add(1)
        self.r2_score.assign_add(1 - tf.divide(ss_res, ss_tot))

    def result(self):
        return tf.divide(self.r2_score, self.n_batchs)
