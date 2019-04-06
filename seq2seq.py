import numpy as np
import tensorflow as tf
import os
from absl import app
from absl import flags
from util import generate_sentece, shuffle_data, pad_sequences,load_test_data,getsentencce, get_hack_data,get_requests_from_file, generate_sentence_int
import time
import math
from random import shuffle
import random

class Ano_Autoencoder(object):
	def __init__(self,is_training,epoch,checkpoint_dir,learning_rate,batch_size,beta1,beta2):
		""""
		Args:
			beta1: beta1 for AdamOptimizer
			beta2: beta2 for AdamOptimizer
			learning_rate: learning_rate for the AdamOptimizer
			training: [bool] Training/NoTraining
			batch_size: size of the batch_
			epoch: number of epochs
			checkpoint_dir: directory in which the model will be saved
		"""

		self.beta1 = beta1
		self.beta2 = beta2
		self.learning_rate = learning_rate
		self.training = is_training
		self.batch_sizer = batch_size
		self.epoch = epoch
		self.checkpoint_dir = checkpoint_dir
		self.save_epoch = 0
		self.hidden_units = 128
		self.vocab_size = 87
		self.num_lstm = 256
		self.build_network()

	def process_decoder_input(self,target_data,char_to_code, batch_size):
		ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
		dec_input = tf.concat([tf.fill([batch_size, 1], char_to_code), ending], 1)
		return dec_input

	def lstm_cell(self,hidden_layer_size):
		cells = tf.nn.rnn_cell.LSTMCell(hidden_layer_size, forget_bias=1.0,reuse=tf.AUTO_REUSE)
		return cells

	def encoder(self,input,seq_lens):
		with tf.variable_scope("encoder") as scope:
			cells = [self.lstm_cell(self.num_lstm) for _ in range(2)]
			x = tf.nn.rnn_cell.MultiRNNCell(cells,state_is_tuple=True)
			rnn_outputs, final_state = tf.nn.dynamic_rnn(x, input, sequence_length=seq_lens,swap_memory=True,dtype=tf.float32)
		return final_state


	def decoder_inference(self,enc_state,embeddings,go_index,eos_index,batch_size):
		helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,tf.fill([batch_size], go_index),eos_index)
		cells = [self.lstm_cell(self.num_lstm) for _ in range(2)]
		dec_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
		decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, enc_state,output_layer=tf.layers.Dense(self.vocab_size,activation=tf.nn.softmax,kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)))
		outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=False,impute_finished=True,maximum_iterations=self.max_seq_len, swap_memory=True)
		logits = tf.identity(outputs.rnn_output, 'logits')

		return logits

	def decoder(self,enc_state,dec_embed_input):
		output_lengths = tf.ones([self.batch_si], tf.int32) * self.max_seq_len
		helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input,output_lengths,time_major=False)
		cells = [self.lstm_cell(self.num_lstm) for _ in range(2)]
		dec_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
		decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, enc_state,output_layer=tf.layers.Dense(self.vocab_size,kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)))
		outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=False,impute_finished=True,maximum_iterations=self.max_seq_len, swap_memory=True)
		logits = tf.identity(outputs.rnn_output, 'logits')

		return logits


	def build_network(self):
		self.input = tf.placeholder(tf.int32, [None,None], name="real_data")
		self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
		self.seq_length = tf.placeholder(tf.float32,[None],name="lengh_of_sequenze")
		self.batch_si = tf.placeholder(tf.int32,[],name="batchsize")
		self.go_index = tf.placeholder(tf.int32,[],name="goindex")
		self.eos_index = tf.placeholder(tf.int32,[],name="endofsentenz")
		self.max_seq_len =  tf.placeholder(tf.int32,[],name='max_target_len')

		with tf.variable_scope('embedding_dec', reuse=tf.AUTO_REUSE):
			self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size , self.hidden_units],-1.0,1.0))
			self.encoder_input = tf.nn.embedding_lookup(self.embeddings, self.input)

		self.encoder_output = self.encoder(self.encoder_input,self.seq_length)

		self.dec_input = self.process_decoder_input(self.targets,self.go_index,self.batch_si)

		with tf.variable_scope('embedding_enc', reuse=tf.AUTO_REUSE):
			self.embeddings_2 = tf.Variable(tf.random_uniform([self.vocab_size , self.hidden_units],-1.0,1.0))
			self.decoder_embedded_input = tf.nn.embedding_lookup(self.embeddings_2, self.dec_input)


		with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
			self.decoder_output = self.decoder(self.encoder_output,self.decoder_embedded_input)

		with tf.variable_scope("decoder",reuse=True):
			self.decoder_output_infer = self.decoder_inference(self.encoder_output,self.embeddings,self.go_index,self.eos_index,self.batch_si)

		#self.decoder_output_infer = tf.Print(self.decoder_output_infer,[tf.shape(self.decoder_output_infer)])
		self.tensor = tf.identity(self.decoder_output)
		self.probs = tf.nn.softmax(self.tensor, name='probs')
		self.masks = tf.sequence_mask(self.seq_length, self.max_seq_len,dtype=tf.float32, name='masks')

		self.cross_entropy_2 = tf.contrib.seq2seq.sequence_loss(self.decoder_output_infer ,self.targets,self.masks,name='cross_entropy_2')
		self.cross_entropy = tf.contrib.seq2seq.sequence_loss(self.decoder_output,self.targets,self.masks,name='cross_entropy')


		self.loss_2 = tf.reduce_mean(self.cross_entropy_2)
		self.loss = tf.reduce_mean(self.cross_entropy)
		self.summary_loss = tf.summary.scalar("loss",self.loss)

		print("init_d_optim")
		self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1,beta2 = self.beta2)
		self.gradients = self.optim .compute_gradients(self.loss)
		self.capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gradients if grad is not None]
		self.optim = self.optim.apply_gradients(self.capped_gradients)


		self.saver = tf.train.Saver()

		#Tensorboard variables
		#self.summary_g_loss = tf.summary.scalar("g_loss",self.g_loss)
		self.summaryloss = tf.summary.scalar("loss",self.loss)


	def save_model(self, iter_time):
		model_name = 'model'
		self.saver.save(self.sess, os.path.join(self.checkpoint_dir, model_name), global_step=iter_time)
		print('=====================================')
		print('             Model saved!            ')
		print('=====================================\n')


	def train(self):
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth = True
		self.sess = tf.Session(config = self.config)
		with self.sess:
			if self.load_model():
				print(' [*] Load SUCCESS!\n')
			else:
				print(' [!] Load Failed...\n')
				self.sess.run(tf.global_variables_initializer())
			train_writer = tf.summary.FileWriter("./logs",self.sess.graph)
			merged = tf.summary.merge_all()
			self.counter = 1
			word2int, int2word,vocab_size,self.training_data =  get_hack_data()
			self.go = word2int["<GO>"]
			self.end= word2int["<EOS>"]
			self.pad= word2int["<PAD>"]
			#print(self.training_data.shape)
			k = (len(self.training_data) // self.batch_sizer)
			self.start_time = time.time()
			loss_g_val,loss_d_val = 0, 0
			self.training_data = self.training_data[0:(self.batch_sizer*k)]
			test_counter = 0
			print('Starting the Training....')
			print(self.end)

			for e in range(0,self.epoch):
				epoch_loss = 0.
				self.training_data = shuffle_data(self.training_data)
				mean_epoch_loss =[]
				for i in range(0,k):
					print(i)
					batch = self.training_data[i*self.batch_sizer:(i+1)*self.batch_sizer]
					length = len(max(batch,key=len))
					batched_data,l = pad_sequences(batch,word2int)
					batched_data = np.asarray(batched_data,dtype="int32")
					_, loss_val, loss_histo= self.sess.run([self.optim,self.loss,self.summary_loss],feed_dict={self.input: batched_data,self.targets: batched_data,self.max_seq_len:length,self.seq_length: l, self.batch_si:self.batch_sizer,self.go_index: self.go})
					train_writer.add_summary(loss_histo,self.counter)
					self.counter=self.counter + 1
					epoch_loss += loss_val
					mean_epoch_loss.append(loss_val)
				mean = np.mean(mean_epoch_loss)
				std = np.std(mean_epoch_loss)
				epoch_loss /= k
				print('Validation loss mean: ', mean)
				print('Validation loss std: ', std)
				print("Loss of Seq2Seq Model: %f" % epoch_loss)
				print("Epoch%d" %(e))

				if e % 1 == 0:
					save_path = self.saver.save(self.sess,"C:/Users/Andreas/Desktop/seq2seq - continous/checkpoint/model.ckpt",global_step=self.save_epoch)
					print("model saved: %s" %save_path)

					data = get_requests_from_file("C:/Users/Andreas/Desktop/seq2seq - continous/data/anomaly.txt")
					random_number = np.random.randint(0,len(data))

					data = generate_sentence_int([data[random_number]],word2int)
					batched_test_data,l = pad_sequences(data,word2int)
					batched_test_data = np.asarray(batched_test_data,dtype="int32")
					ba_si=1
					size = l[0]
					print(batched_test_data)
					w,test,loss_eval= self.sess.run([self.probs,self.decoder_output,self.loss],feed_dict={self.input: batched_test_data,self.max_seq_len:size ,self.seq_length: l, self.batch_si: ba_si,self.go_index: self.go, self.eos_index: self.end,self.targets: batched_test_data})

					coefs = np.array([w[j][batched_test_data[0][j]] for j in range(len(batched_test_data))])
					print(coefs)
					coefs = coefs / coefs.max()
					print(coefs)
					print(coefs.shape)
					intsent = np.argmax(test,axis=2)
					tester = getsentencce(intsent[0],int2word)
					print(tester)
					self.save_epoch += 1
					print("Loss of test_data: %f" %  loss_eval)

			print("training finished")


	def load_model(self):
		print(' [*] Reading checkpoint...')
		ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
			meta_graph_path = ckpt.model_checkpoint_path + '.meta'
			self.save_epoch = int(meta_graph_path.split('-')[-1].split('.')[0])
			print('===========================')
			print('   iter_time: {}'.format(self.save_epoch))
			print('===========================')
			return True
		else:
			return False
