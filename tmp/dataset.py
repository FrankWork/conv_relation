import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset.output_types)  # ==> "tf.float32"
print(dataset.output_shapes)  # ==> "(10,)"

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  print(sess.run(next_element))