import tensorflow as tf

y_true = tf.constant([[5.2, 0, 0, 1.5]])
mask = tf.cast((y_true > 0), dtype=tf.float32)
print(mask)