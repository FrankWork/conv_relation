from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os
import time
import sys
import tensorflow as tf
import numpy as np
from reader import base as base_reader
from models import cnn_model
from models import mtl_model

# tf.set_random_seed(0)
# np.random.seed(0)

flags = tf.app.flags

flags.DEFINE_string("train_file", "data/train.txt", 
                             "original training file")
flags.DEFINE_string("test_file", "data/test.txt", 
                             "original test file")

flags.DEFINE_string("vocab_file", "data/vocab.txt", 
                              "vocab of train and test data")

flags.DEFINE_string("google_embed300_file", 
                             "data/embed300.google.npy", 
                             "google news word embeddding")
flags.DEFINE_string("google_words_file", 
                             "data/google_words.lst", 
                             "google words list")
flags.DEFINE_string("trimmed_embed300_file", 
                             "data/embed300.trim.npy", 
                             "trimmed google embedding")

flags.DEFINE_string("senna_embed50_file", 
                             "data/embed50.senna.npy", 
                             "senna words embeddding")
flags.DEFINE_string("senna_words_file", 
                             "data/senna_words.lst", 
                             "senna words list")
flags.DEFINE_string("trimmed_embed50_file", 
                             "data/embed50.trim.npy", 
                             "trimmed senna embedding")

flags.DEFINE_string("train_record", "data/train.tfrecord", 
                             "training file of TFRecord format")
flags.DEFINE_string("test_record", "data/test.tfrecord", 
                             "Test file of TFRecord format")
flags.DEFINE_string("train_mtl_record", "data/train.mtl.tfrecord", 
                             "Multi-task learing training file of TFRecord format")
flags.DEFINE_string("test_mtl_record", "data/test.mtl.tfrecord", 
                             "Multi-task learing test file of TFRecord format")

flags.DEFINE_string("relations_file", "data/relations_new.txt", "relations file")
flags.DEFINE_string("results_file", "data/results.txt", "predicted results file")
flags.DEFINE_string("logdir", "saved_models/", "where to save the model")

# flags.DEFINE_integer("freq_threshold", None, "vocab frequency threshold to keep the word")
flags.DEFINE_integer("max_len", 96, "max length of sentences")
flags.DEFINE_integer("num_relations", 19, "number of relations")
flags.DEFINE_integer("word_dim", 50, "word embedding size")
flags.DEFINE_integer("num_epochs", 200, "number of epochs")
flags.DEFINE_integer("batch_size", 100, "batch size")

flags.DEFINE_integer("pos_num", 123, "number of position feature")
flags.DEFINE_integer("pos_dim", 5, "position embedding size")
# flags.DEFINE_integer("filter_size", 3, "cnn number of hidden unit")
flags.DEFINE_integer("num_filters", 100, "cnn number of output unit")
# flags.DEFINE_integer("linear_size", 200, "linear layer number of hidden unit")
flags.DEFINE_float("loss_diff_coef", 0.000001, "coefficient of Orthogonality Constraints")

# flags.DEFINE_integer("rnn_size", 100, "hidden unit of rnn")
# flags.DEFINE_integer("rnn_layers", 1, "layers of rnn")

# flags.DEFINE_integer("decay_steps", 1*80, "learning rate decay steps")
# flags.DEFINE_float("decay_rate", 0.97, "learning rate decay rate")
flags.DEFINE_float("lrn_rate", 1e-3, "learning rate")
flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")

flags.DEFINE_string("model", "cnn", "cnn or mtl model")
flags.DEFINE_boolean('test', False, 'set True to test')
flags.DEFINE_boolean('trace', False, 'set True to test')

FLAGS = tf.app.flags.FLAGS

def trace_runtime(sess, m_train):
  '''
  trace runtime bottleneck using timeline api

  navigate to the URL 'chrome://tracing' in a Chrome web browser, 
  click the 'Load' button and locate the timeline file.
  '''
  run_metadata=tf.RunMetadata()
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  from tensorflow.python.client import timeline
  trace_file = open('timeline.ctf.json', 'w')

  fetches = [m_train.train_op, m_train.loss, m_train.accuracy]
  _, loss, acc = sess.run(fetches, 
                            options=options, 
                            run_metadata=run_metadata)
                            
  trace = timeline.Timeline(step_stats=run_metadata.step_stats)
  trace_file.write(trace.generate_chrome_trace_format())
  trace_file.close()


def train(sess, m_train, m_valid):
  n = 1
  best = .0
  best_step = n
  start_time = time.time()
  orig_begin_time = start_time

  fetches = [m_train.train_op, m_train.loss, m_train.accuracy]

  while True:
    try:
      _, loss, acc = sess.run(fetches)

      epoch = n // 80
      if n % 80 == 0:
        now = time.time()
        duration = now - start_time
        start_time = now
        v_acc = sess.run(m_valid.accuracy)
        if best < v_acc:
          best = v_acc
          best_step = n
          m_train.save(sess, best_step)
        print("Epoch %d, loss %.2f, acc %.2f %.4f, time %.2f" % 
                                  (epoch, loss, acc, v_acc, duration))
        sys.stdout.flush()
      n += 1
    except tf.errors.OutOfRangeError:
      break

  duration = time.time() - orig_begin_time
  duration /= 3600
  print('Done training, best_step: %d, best_acc: %.4f' % (best_step, best))
  print('duration: %.2f hours' % duration)
  sys.stdout.flush()

def test(sess, m_valid):
  m_valid.restore(sess)
  fetches = [m_valid.accuracy, m_valid.prediction]
  accuracy, predictions = sess.run(fetches)
  print('accuracy: %.4f' % accuracy)
  
  base_reader.write_results(predictions, FLAGS.relations_file, FLAGS.results_file)


def main(_):
  with tf.Graph().as_default():
    train_data, test_data, word_embed = base_reader.inputs()

    # sv = tf.train.Supervisor()
    # with sv.managed_session() as sess:
    #   print('='*80)
    #   for i in range(10):
    #     arr = sess.run(test_data)
    #     print(train_data[2].shape, arr[2].shape)
    #     print(arr[2][0])
    #   exit()

    
    if FLAGS.model == 'cnn':
      m_train, m_valid = cnn_model.build_train_valid_model(word_embed, 
                                                      train_data, test_data)
    elif FLAGS.model == 'mtl':
      m_train, m_valid = mtl_model.build_train_valid_model(word_embed, 
                                                      train_data, test_data)
    
    m_train.set_saver(FLAGS.model)
    
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())# for file queue

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存 
    config.gpu_options.allow_growth = True
    
    # sv finalize the graph
    with tf.Session(config=config) as sess:
      sess.run(init_op)
      print('='*80)

      if FLAGS.trace:
        trace_runtime(sess, m_train)
      elif FLAGS.test:
        test(sess, m_valid)
      else:
        train(sess, m_train, m_valid)

      
  
          
if __name__ == '__main__':
  tf.app.run()
