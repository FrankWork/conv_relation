# TODO: reduce vocab size
MAX_VOCAB_SIZE = None # 9006 + 1
EOS_TOKEN = '</s>'


# Data filenames
# Sequence Autoencoder
ALL_SA = 'all_sa.tfrecords'
TRAIN_SA = 'train_sa.tfrecords'
TEST_SA = 'test_sa.tfrecords'
# Language Model
ALL_LM = 'all_lm.tfrecords'
TRAIN_LM = 'train_lm.tfrecords'
TEST_LM = 'test_lm.tfrecords'
# Classification
TRAIN_CLASS = 'train_cl.tfrecords'
TEST_CLASS = 'test_cl.tfrecords'
# LM with bidirectional LSTM
TRAIN_REV_LM = 'train_reverse_lm.tfrecords'
TEST_REV_LM = 'test_reverse_lm.tfrecords'
# Classification with bidirectional LSTM
TRAIN_BD_CLASS = 'train_bidir_cl.tfrecords'
TEST_BD_CLASS = 'test_bidir_cl.tfrecords'

