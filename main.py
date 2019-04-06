import numpy as np
import tensorflow as tf
import os
from absl import app
from absl import flags
from seq2seq import Ano_Autoencoder
import sys

flags = tf.app.flags

flags.DEFINE_integer("epochs",1000,"epochs per trainingstep")
flags.DEFINE_float("learning_rate",0.0001,"learning rate for the model")
flags.DEFINE_bool("training",True,"running training of the poincloud gan")
flags.DEFINE_string("checkpoint_dir","C:/Users/Andreas/Desktop/seq2seq - continous/checkpoint","where to save the model")
flags.DEFINE_string("sample_dir","samples","where the samples are stored")
flags.DEFINE_integer("iterations",100000,"number of patches")
flags.DEFINE_integer("batch_size",32,"size of the batch")
flags.DEFINE_float("beta1",0.5,"adam beta1")
flags.DEFINE_float("beta2",0.9,"adam beta2")
FLAGS = flags.FLAGS

def _main(argv):
	print("initializing Params")
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)
	if FLAGS.training == True:
			autoenc = Ano_Autoencoder(FLAGS.training,FLAGS.epochs,FLAGS.checkpoint_dir,FLAGS.learning_rate,FLAGS.batch_size,FLAGS.beta1,FLAGS.beta2)
			autoenc.train()
	else:
		if not cgan.load(FLAGS.checkpoint_dir):
			print("first train your model")

if __name__ == '__main__':
	print('Starting the Programm....')
	app.run(_main)
