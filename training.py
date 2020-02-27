import tensorflow as tf
import argparse, os, pickle
from module.helper import split_data, train_normalization, test_normalization
from module.network import BAPA
import numpy as np

def main():
	
	parser = argparse.ArgumentParser(
		description = "Predict binding affinity with two features for given complex.\
		First feature is the number of occurrences of each descriptor. Second feature is Vina terms.\
		The BAPA model consists of three types of neural networks and is designed to predict protein-ligand \
		binding affinity."
	)	
	parser.add_argument("--dataset", "-d", required = True, type = input_check, nargs = "+", 
			help = "Data containing the number of occurrences of each descriptor\
			calculated for given complex.")
	parser.add_argument("--Vina_terms", "-v", required = True, type = input_check, nargs = "+",
			help = "Data for Vina terms calculated for given complex.")
	parser.add_argument("--save_dir", "-s", required = True, type = output_check, nargs = "+",
			help = "The path where the trained model will be stored.")
	parser.add_argument("--normalized_parmeters", type = input_check, nargs = "+",
			help = "Parameters to use for normalization. This information is necessary for the test model \
			when conducting a test with a trained model.")				
	parser.add_argument("--validation_dataset", type = input_check, nargs = "+",
			help = "The number of occurrences of each descriptor for validation.")	
	parser.add_argument("--validation_Vina_terms", type = input_check, nargs = "+",
			help = "Vina terms to be used for validation..")		

	args = parser.parse_args()

	if len(args.dataset) != 1 or len(args.Vina_terms) != 1:
		raise IOError("Please input only one file related to dataset and Vina terms.")
	data_path, Vina_terms_path = args.dataset[0], args.Vina_terms[0]	
	
	if args.validation_dataset == None and args.validation_Vina_terms == None:
		exist_validation_data = 0
	elif len(args.validation_dataset) == 1 and len(args.validation_Vina_terms) == 1:
		validation_data_path, validation_Vina_terms_path = args.validation_dataset[0], args.validation_Vina_terms[0]
		exist_validation_data = 1
	else:
		raise IOError("Please input only one file related to dataset and Vina terms for validation.")
	
	if args.normalized_parmeters == None:
		normalized_parmeters_path = os.path.abspath("./module/train_normalized_parmeters.pkl")
	elif args.normalized_parmeters != None and len(args.normalized_parmeters) != 1:
		raise IOError("Please input only one file path for save the normalized parmeters.")
	elif args.normalized_parmeters != None and len(args.normalized_parmeters) == 1:
		normalized_parmeters_path = args.normalized_parmeters[0]
		
	if len(args.save_dir) != 1 :
		raise IOError("Please input only one file path for trained the BAPA model.")
	save_dir = args.save_dir[0]	

	print("===============================================================================")
	if exist_validation_data:
		print("[Start] The following input files are being preprocessing ...")
		print("\t%s"%data_path)
		print("\t%s"%Vina_terms_path)
		print("\t%s"%validation_data_path)
		print("\t%s"%validation_Vina_terms_path)
		
		train_names, train_data, train_Vina_terms, train_labels = preprocessing(data_path, Vina_terms_path)
		print("\n\t> %s train complex are ready to predict\n"%(len(names)))
		
		validation_names, validation_data, validation_Vina_terms, validation_labels = preprocessing(validation_data_path, validation_Vina_terms_path)
		print("\n\t> %s Validation complex are ready to predict\n"%(len(names)))
		
		''' nomalized each data '''
		train_data, train_Vina_terms, data_mean, data_std, Vina_mean, Vina_std = train_normalization(train_data, train_Vina_terms, normalized_parmeters_path)
		validation_data, validaton_Vina_terms = test_normalization(validation_data, validation_Vina_terms)
		
	else:
		print("[Start] Validation data does not exist, 80% of train data is used as validation data...")
		print("\t%s"%data_path)
		print("\t%s"%Vina_terms_path)
		
		names, data, Vina_terms, labels = preprocessing(data_path, Vina_terms_path)
		
		''' split train data '''
		train_name, train_data, train_Vina_terms, train_labels, \
			validation_name, validation_data, validation_Vina_terms, validation_labels = split_data(names, data, Vina_terms, labels)
		
		print("\n\t> Total of %s complexes were split into train data(%s) and vlidation data(%s)\n."%(len(names), len(train_name), len(validation_name)))
		
		''' nomalized each data '''
		train_data, train_Vina_terms, data_mean, data_std, Vina_mean, Vina_std = train_normalization(train_data, train_Vina_terms, normalized_parmeters_path)
		validation_data, validaton_Vina_terms = test_normalization(validation_data, validation_Vina_terms)
	
	print("[Load] BAPA train iter...")
	''' load BAPA train iter '''
	BAPA_ = BAPA(num_descriptors = 2500, num_epochs = 500, embedding_size = 10, learning_rate = 0.005, batch_size = 256, save_dir = save_dir)
	BAPA_.get_parameter()
	
	''' run train '''
	BAPA_.run((train_name, train_data, train_Vina_terms, train_labels), 
		(validation_name, validation_data, validaton_Vina_terms, validation_labels))
	
	print("[Finish] BAPA model train finish ...")	
	print("===============================================================================")
	
	
def preprocessing(data_path, Vina_path):		
	
	with open(data_path, "rb") as f:
		input_data, input_labels = pickle.load(f)
	
	with open(Vina_path, "rb") as f:
		input_Vina = pickle.load(f)
	
	names, data, Vina, labels = list(), list(), list(), list()

	for key in input_data.keys():
		names.append(key)
		labels.append(input_labels[key])
		data.append(input_data[key])
		
		#key = key[:4] + "_remove_water/" + key[:4] + "_ligand"
		key = key[:12] + key[20:]
		
		Vina.append(input_Vina[key])
	
	return np.array(names), np.array(data, dtype = np.float32), np.array(Vina, dtype = np.float32), np.array(labels, dtype = np.float32)

def output_check(path):
		
	path = os.path.abspath(path)
	dir_path = os.path.dirname(path)
	if not os.access(dir_path, os.W_OK):
		raise IOError("%s cannot be created."%path) 
	return path

def input_check(path):

	path = os.path.abspath(path)
	if not os.path.exists(path):
		raise IOError('%s does not exist.' %path)
	return path		
	
if __name__ == "__main__":
	main()