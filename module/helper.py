from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import pickle, os
import numpy as np

def get_descriptors_index(N_data, num_descriptors = 2500):
	return np.array([[j for j in range(num_descriptors)] for i in range(N_data)])

def test_normalization(input_data, input_Vina):
	
	normalized_data, normalized_vina = list(), list()

	with open(os.path.abspath("./module/normalized_parameters.pkl"),"rb") as f:
		data_mean, data_std, Vina_mean, Vina_std = pickle.load(f)
	
	for idx, val in enumerate(input_data.T):
		normalized_data.append((val - data_mean[idx]) / data_std[idx]) if data_std[idx] != 0. else normalized_data.append(val)

	for idx, val in enumerate(input_Vina.T):
		normalized_vina.append((val - Vina_mean[idx]) / Vina_std[idx]) if Vina_std[idx] != 0. else normalized_vina.append(val)
	
	return np.array(normalized_data).T, np.array(normalized_vina).T

def train_normalization(input_data, input_Vina, save_path):

	data_mean, data_std, normalized_data = list(), list(), list()
	Vina_mean, Vina_std, normalized_Vina = list(), list(), list()

	for idx, val in enumerate(input_data.T):
		mean_, std_ = np.mean(val), np.std(val)
		
		data_mean.append(mean_)
		data_std.append(std_)
		
		normalized_data.append((val - mean_) / std_) if std_ != 0. else normalized_data.append(val)
	
	for idx, val in enumerate(input_Vina.T):
		mean_, std_ = np.mean(val), np.std(val)

		Vina_mean.append(mean_)
		Vina_std.append(std_)		
		
		normalized_Vina.append((val - mean_) / std_) if std_ != 0. else normalized_Vina.append(val)

	with open(save_path, "wb") as f:
		pickle.dump((data_mean, data_std, Vina_mean, Vina_std),f) 
	
	return np.array(normalized_data).T, np.array(normalized_Vina).T, data_mean, data_std, Vina_mean, Vina_std

def split_data(input_names, input_data, input_Vina, input_labels):	
	
	idx = np.arange(0 , len(input_names))
	np.random.shuffle(idx)
	train_idx, validation_idx = idx[:int(len(input_names)*0.8)], idx[int(len(input_names)*0.8):]
	
	train_names, train_data, train_Vina, train_labels = get_data(train_idx, input_names, input_data, input_Vina, input_labels)
	validation_names, validation_data, validation_Vina, validation_labels = get_data(validation_idx, input_names, input_data, input_Vina, input_labels)
	
	return train_names, train_data, train_Vina, train_labels, \
		validation_names, validation_data, validation_Vina, validation_labels
	
def get_data(index, input_names, input_data, input_Vina, input_labels):	
	return np.array([input_names[i] for i in index]), np.array([input_data[i] for i in index], dtype = np.float32), \
		np.array([input_Vina[i] for i in index], dtype = np.float32), np.array([input_labels[i] for i in index], dtype = np.float32)

def get_results(labels, predictions):

	RMSE = np.sqrt(mean_squared_error(labels.flatten(), predictions.flatten()))
	MAE = mean_absolute_error(labels.flatten(), predictions.flatten())
	PCC = pearsonr(labels.flatten(), predictions.flatten())	
	
	regr = linear_model.LinearRegression()
	regr.fit(predictions, labels)
	testpredy = regr.predict(predictions)

	testmse = mean_squared_error(labels.flatten(), testpredy.flatten())
	num = labels.shape[0]
	SD = np.sqrt((testmse * num) / (num -1))
	
	return RMSE, MAE, PCC[0], SD		
