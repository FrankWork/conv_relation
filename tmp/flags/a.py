import tensorflow as tf
import b

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('a_param', 'in a', 'Which dataset to generate data for')

def main(_):
  if FLAGS.a_param:
    print(FLAGS.a_param)
  if FLAGS.b_param:
    print(FLAGS.b_param)

if __name__ == '__main__':
  tf.app.run()
