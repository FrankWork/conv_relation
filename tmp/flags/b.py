import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('b_param', 'in b', 'Which dataset to generate data for')