import tensorflow as tf
import numpy as np
import argparse, os, pickle
from module.helper import get_descriptors_index, test_normalization

def main():

	parser = argparse.ArgumentParser(
		description = "Predict binding affinity with two features for given complex.\
		First feature is the number of occurrences of each descriptor. Second feature is Vina terms.\
		Note: If you want to use the newly trained BAPA model, you need to modify and use \
		'nomalized_parameters.pkl' and 'network's information'"
	)
	
	parser.add_argument("--dataset", "-d", required = True, type = input_check, nargs = "+",
			help = "Data containing the number of occurrences of each descriptor\
			calculated for given complex.")
	parser.add_argument("--Vina_terms", "-v", required = True, type = input_check, nargs = "+",
			help = "Data for Vina terms calculated for given complex.")
	parser.add_argument("--output_file", "-o", required = True, type = output_check, nargs = "+",
			help = "File to save the model's predictions for given complex.")
		
	args = parser.parse_args()

	if len(args.dataset) != 1 or len(args.Vina_terms) != 1:
		raise IOError("Please input only one file for data and one file for Vina terms.")
	data_path, Vina_terms_path = args.dataset[0], args.Vina_terms[0]

	if len(args.output_file) != 1 :
		raise IOError("Please input only one file path for save the BAPA predictions.")
	output_file = args.output_file[0]	 
	
	print("===============================================================================")
	print("Nont: If you want to use the newly trained BAPA model, you need to ")
	print("[Start] The following input files are being preprocessing ...") 
	print("\t%s"%data_path)
	print("\t%s"%Vina_terms_path)
	names, data, Vina_terms = preprocessing(data_path, Vina_terms_path)
	
	''' nomalized each data '''
	data, Vina_terms = test_normalization(data, Vina_terms)
	print("\n\t> %s complex are ready to predict\n"%(len(names)))
	
	''' run test '''
	print("[Prediction] The BAPA model is making predictions, so please wait a moment ...")
	run_test(names, data, Vina_terms, output_file)
	print("[Finish] %s output file is generated ..." %output_file)
	print("===============================================================================")
	
def run_test(names, data, Vina_terms, output_file):
	
	SAVER_DIR = os.path.abspath("./module/network")
	tf.reset_default_graph()

	descriptors_index = get_descriptors_index(len(data), 2500)	
	
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(SAVER_DIR + '/BAPA_network.meta')
		ckpt = tf.train.get_checkpoint_state(SAVER_DIR)
		saver.restore(sess, ckpt.model_checkpoint_path)
		
		''' load variables  '''
		DESCRIPTORS_INDEX, DESCRIPTION_MATRIX, VINA_TERMS, _, keep_prob = tf.get_collection("input_variables")
		predicted_variables = tf.get_collection("predict")
		
		''' run predictions '''
		predictions = sess.run(predicted_variables[1], 
			feed_dict = {DESCRIPTORS_INDEX:descriptors_index, DESCRIPTION_MATRIX:data, VINA_TERMS:Vina_terms, keep_prob:1.0})
			
		fwrite(open(output_file, "w"), names, predictions.flatten())	
		
def fwrite(fw, names, predictions):	
	
	fw.write("complex,predictions\n")
	
	for name, prediction in zip(names, predictions):
		fw.write("%s,%.2f\n"%(name, prediction))
	fw.close()
	
def preprocessing(data_path, Vina_path):		
	
	with open(data_path, "rb") as f:
		input_data = pickle.load(f)
	
	with open(Vina_path, "rb") as f:
		input_Vina = pickle.load(f)
	
	names, data, Vina = list(), list(), list()

	for key in input_data.keys():
		names.append(key)
		data.append(input_data[key])
		
		#key = key[:4] + "_remove_water/" + key[:4] + "_ligand"
		#key = key[:12] + key[20:]
		
		Vina.append(input_Vina[key])
	
	return np.array(names), np.array(data, dtype = np.float32), np.array(Vina, dtype = np.float32)

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
	
def read_file(file):
	return file.readlines()
	
if __name__ == "__main__":
	main()
