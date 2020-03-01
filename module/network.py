import tensorflow as tf
import numpy as np
from module.helper import get_descriptors_index, get_results

class BAPA():
	
	def __init__(self, num_descriptors, num_epochs, embedding_size, learning_rate, batch_size,
				save_dir, save_flag = False, save_step = 10):
		
		self.num_descriptors = num_descriptors
		self.num_epochs = num_epochs
		self.embedding_size = embedding_size 
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-500)
		self.save_flag = save_flag
		self.save_dir = save_dir
		self.save_step = save_step
		
	def get_parameter(self):
		
		print("> Train parameters ...")
		print("\tnum epochs: %s"%(self.num_epochs))
		print("\tbatch size: %s"%(self.batch_size))
		print("\tembedding size: %s"%(self.embedding_size))
		print("\tlearning rate: %s"%(self.learning_rate))
		
		print("\ttrained model is stored evey %s steps."%self.save_step) if self.save_flag else print("\ttrained model is not saved")

	def run(self, train_tuple, validation_tuple):
		
		print("[RUN] BAPA trainiter")
		
		train_names, train_data, train_Vina, train_labels = \
			train_tuple[0], train_tuple[1], train_tuple[2], train_tuple[3]
			
		validation_names, validation_data, validation_Vina, validaiton_labels = \
			validation_tuple[0], validation_tuple[1], validation_tuple[2], validation_tuple[3]
		
		tf.reset_default_graph()
		self.global_step = tf.Variable(0, name = "global_step", trainable=False)
		
		print("\tCreate train model")
		''' model defined '''	
		self.predictions = self.make_model()
		descriptors_index = get_descriptors_index(N_data = self.batch_size, num_descriptors = self.num_descriptors)
		
		''' add input placeholder '''
		for op in [self.DESCRIPTORS_INDEX, self.DESCRIPTION_MATRIX, self.VINA_TERMS, self.input_labels, self.keep_prob]:
			tf.add_to_collection("input_variables",op)		
		
		''' loss defined '''
		with tf.variable_scope("loss") as scope:
			print("\tCreate loss function")
			self.labels = tf.cast(tf.expand_dims(self.input_labels,1), tf.float32)
			self.loss = tf.reduce_mean(tf.pow((self.labels - self.predictions),2))
			
			tf.add_to_collection("regression_loss",self.loss)
			
		loss = tf.add_n(tf.get_collection("regression_loss"))

		if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
			loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))	
		
		''' add predictions variable '''
		for op in [self.labels, self.predictions, self.loss, 'self.images', self.embedding_matrix, 
			self.embeddings, self.embeddings_, self.conv_emb_, self.encoded_vector, self.weights, 
				self.attention_vector, self.weights_dot_attention_vector, self.context_vector]:
			tf.add_to_collection("predict",op)	
		
		''' optimizer '''
		train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step = self.global_step)	
		
		if self.save_flag:
			saver = tf.train.Saver(max_to_keep = None)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			
			wasKeyboardInterrupt = False

			i = 0
			try:			
				while i < self.num_epochs:
					
					''' training '''
					batch = self.get_next_batch(train_data, train_Vina, train_labels)
			
					_, batch_loss, batch_labels, batch_predictions = sess.run([train_step, self.loss, self.labels, self.predictions],
						feed_dict = {self.DESCRIPTORS_INDEX:descriptors_index, self.DESCRIPTION_MATRIX:batch[0], 
							self.VINA_TERMS:batch[1], self.input_labels:batch[2], self.keep_prob:1.0})
					
					RMSE, MAE, PCC, SD = get_results(batch_labels, batch_predictions)
					print("%s iteration batch loss: %.2f, RMSE: %.2f, MAE: %.2f, PCC: %.2f, SD: %.2f"%(i, batch_loss, RMSE, MAE, PCC, SD))
					
					''' validation '''
					loss_validation, labels_validation, predictions_validation = sess.run([self.loss, self.labels, self.predictions], 
						feed_dict = {self.DESCRIPTORS_INDEX:get_descriptors_index(N_data = len(validation_names), num_descriptors = self.num_descriptors), 
							self.DESCRIPTION_MATRIX:validation_data, self.VINA_TERMS:validation_Vina, 
								self.input_labels:validaiton_labels, self.keep_prob:1.0})
					

					RMSE, MAE, PCC, SD = get_results(labels_validation, predictions_validation)
					print("%s iteration validation loss: %.2f, RMSE: %.2f, MAE: %.2f, PCC: %.2f, SD: %.2f"%(i, loss_validation, RMSE, MAE, PCC, SD))
				
					if self.save_flag:
						saver.save(sess, self.save_dir + "/model", global_step = i)	
						
					i += 1
				
			except KeyboardInterrupt as err:
				print('Training interrupted at %d' % i)
				wasKeyboardInterrupt = True	
				raisedEx = err	
				
			finally:
				if wasKeyboardInterrupt:
					raise raisedEx				
			
	def init_placeholder(self):
		''' for placeholder '''
		self.DESCRIPTORS_INDEX = tf.placeholder(shape = (None, self.num_descriptors),  dtype = tf.int32,
			name = 'description_index'
		)
		self.DESCRIPTION_MATRIX = tf.placeholder(shape = (None, self.num_descriptors), dtype = tf.float32,
			name = 'description_matrix'
		)
		self.VINA_TERMS = tf.placeholder(shape = (None, 6), dtype = tf.float32,
			name = 'Vina_terms'
		)
		self.keep_prob = tf.placeholder(
			dtype = "float"
		)
		self.input_labels = tf.placeholder(shape = (None), dtype = tf.float32, 
			name = "input_labels"
		)	
			
	def make_attention_layer(self, X, name = None):
	
		with tf.variable_scope(name, default_name = "Attention_layer") as scope:
			
			h1 = tf.expand_dims(tf.layers.dense(inputs = X, units = self.embedding_size, activation = tf.nn.tanh, 
				kernel_initializer = tf.truncated_normal_initializer(stddev=5e-2), kernel_regularizer = self.regularizer), 1)
			h2 = tf.transpose(tf.reshape(tf.layers.dense(inputs = tf.reshape(self.embeddings, (-1, self.embedding_size)), 
				units = self.embedding_size, activation = tf.nn.tanh, kernel_initializer = tf.truncated_normal_initializer(stddev=5e-2), 
					kernel_regularizer = self.regularizer), (-1, self.num_descriptors, self.embedding_size)), perm = (0, 2, 1))

			self.weights = tf.nn.dropout(tf.nn.elu(tf.reshape(tf.matmul(h1, h2), (-1, self.num_descriptors))), self.keep_prob)
			self.attention_vector = self.softmax_layer(self.weights)
			
			self.weights_dot_attention_vector = tf.reshape(tf.multiply(tf.reshape(self.attention_vector, [-1, 1]), tf.reshape(self.embeddings,[-1, self.embedding_size])), 
				[-1, self.num_descriptors, self.embedding_size])
				
			self.context_vector = tf.reduce_sum(self.weights_dot_attention_vector, axis = 1)
			
			return self.context_vector		
			
	def softmax_layer(self, data, axis = 1):
		
		with tf.variable_scope('Softmax') as scope:
			data_exp = tf.exp(data)
			sum_value = tf.reduce_sum(data_exp, axis = 1, keepdims = True)
			prob = tf.div(data_exp, sum_value)
			
			return prob	

	def make_conv_pool_layer(self, zip_):
		
		conv_pool = tf.layers.conv2d(inputs = self.input_matrix, filters = zip_[0][0], kernel_size = zip_[0][1], padding = "VALID", 
			activation = tf.nn.elu, kernel_regularizer = self.regularizer)
		conv_pool = tf.layers.max_pooling2d(inputs = conv_pool, pool_size = zip_[0][2], strides = zip_[0][2], padding = "VALID") 
		
		for val in zip_[1:]:
			conv_pool = tf.layers.conv2d(inputs = conv_pool, filters = val[0], kernel_size = val[1], padding = "VALID", 
				activation = tf.nn.elu, kernel_regularizer = self.regularizer)
			conv_pool = tf.layers.max_pooling2d(inputs = conv_pool, pool_size = val[2], strides = val[2], padding = "VALID") 		
	
		return conv_pool			

	def make_model(self):
		with tf.variable_scope('whole') as scope:
			
			self.init_placeholder()
			self.embedding_()
		
			with tf.variable_scope('model') as scope:
				
				''' First conv-pool layer '''
				i_1 = self.make_conv_pool_layer([(3, (2, 10), (2, 1)), (6, (2, 1), (4, 1)), (6, (2, 1), (4, 1)), (9, (2, 1), (6, 1))])
				''' Second conv-pool layer '''
				i_2 = self.make_conv_pool_layer([(3, (4, 10), (2, 1)), (6, (4, 1), (4, 1)), (6, (4, 1), (4, 1)), (9, (4, 1), (6, 1))])		
				''' Third conv-pool layer '''
				i_3 = self.make_conv_pool_layer([(3, (6, 10), (2, 1)), (6, (6, 1), (4, 1)), (6, (6, 1), (4, 1)), (9, (6, 1), (6, 1))])							
				''' Four conv-pool layer '''
				i_4 = self.make_conv_pool_layer([(3, (8, 10), (2, 1)), (6, (8, 1), (4, 1)), (6, (8, 1), (4, 1)), (9, (8, 1), (6, 1))])		
				''' Five conv-pool layer '''
				i_5 = self.make_conv_pool_layer([(3, (10, 10), (2, 1)), (6, (10, 1), (4, 1)), (6, (10, 1), (4, 1)), (9, (10, 1), (6, 1))])
				
				''' Concatenate '''
				conv_emb_ = tf.concat([tf.layers.flatten(i_1), tf.layers.flatten(i_2), tf.layers.flatten(i_3), tf.layers.flatten(i_4), tf.layers.flatten(i_5)],1)
				
				''' Block of dense layer '''
				conv_emb_ = tf.layers.dense(inputs = conv_emb_, units = 10, activation = tf.nn.elu, kernel_regularizer = self.regularizer)
				self.conv_emb_ = tf.nn.dropout(conv_emb_, self.keep_prob)
				
				emb_ = tf.concat([self.conv_emb_, self.VINA_TERMS],1)
				emb_ = tf.layers.dense(inputs = emb_, units = 128, activation = tf.nn.elu, kernel_regularizer = self.regularizer)
				emb_ = tf.nn.dropout(emb_, self.keep_prob)
				
				emb_ = tf.layers.dense(inputs = emb_, units = 10, activation = tf.nn.elu, kernel_regularizer = self.regularizer)
				self.encoded_vector = tf.nn.dropout(emb_, self.keep_prob)
				
				''' Attention layer '''
				context_vector = self.make_attention_layer(self.encoded_vector)

				encoded_context_vector = tf.layers.dense(inputs = context_vector, units = 10, activation = tf.nn.elu, kernel_regularizer = self.regularizer)
				encoded_context_vector = tf.nn.dropout(encoded_context_vector, self.keep_prob)
				
				''' Last dense layer '''
				complex_info = tf.layers.dense(inputs = tf.concat([self.encoded_vector, encoded_context_vector], 1), units = 512, 
					activation = tf.nn.elu, kernel_regularizer = self.regularizer)			
				complex_info = tf.nn.dropout(complex_info, self.keep_prob)

				complex_info = tf.layers.dense(inputs = complex_info, units = 256, activation = tf.nn.elu, kernel_regularizer = self.regularizer)
				complex_info = tf.nn.dropout(complex_info, self.keep_prob)

				complex_info = tf.layers.dense(inputs = complex_info, units = 128, activation = tf.nn.elu, kernel_regularizer = self.regularizer)
				complex_info = tf.nn.dropout(complex_info, self.keep_prob)
				
				complex_info = tf.layers.dense(inputs = complex_info, units = 1)
				
				return complex_info
				
	def get_next_batch(self, data, Vina_terms, labels):
		
		idx = np.arange(0 , len(data))
		np.random.shuffle(idx)
		idx = idx[:self.batch_size]
		data_shuffle = [data[i] for i in idx]
		add_shuffle = [Vina_terms[i] for i in idx]
		labels_shuffle = [labels[i] for i in idx]

		return np.asarray(data_shuffle), np.asarray(add_shuffle), np.asarray(labels_shuffle)				 

	def embedding_(self):
		
		''' for embedding '''
		initializer = tf.truncated_normal_initializer(stddev = 1e-1)
		
		self.embedding_matrix = tf.get_variable(
			name = "embedding_matrix",
			shape = [self.num_descriptors, self.embedding_size],
			initializer = initializer,
			dtype = tf.float32
		)

		self.embeddings = tf.nn.embedding_lookup(self.embedding_matrix, self.DESCRIPTORS_INDEX)
		print(self.embeddings)
		self.embeddings_ = tf.multiply(tf.transpose(self.embeddings, perm = [0,2,1]), tf.expand_dims(self.DESCRIPTION_MATRIX, 1))
	
		self.input_matrix = tf.reshape(tf.transpose(tf.nn.dropout(
				tf.reshape(tf.layers.dense(inputs = tf.reshape(self.embeddings_, (-1, self.num_descriptors)), 
					units = self.num_descriptors, activation = tf.nn.elu , kernel_initializer = tf.truncated_normal_initializer(stddev=5e-2), 
						kernel_regularizer = self.regularizer), (-1, self.embedding_size, self.num_descriptors)), 
							self.keep_prob), perm=[0,2,1]), [-1, self.num_descriptors, self.embedding_size, 1])
