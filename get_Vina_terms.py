import numpy as np
import argparse, os, pickle

def main():
	
	parser = argparse.ArgumentParser(
		description = "This script takes in the pdbqt files of the protein and ligand \
		and calculates the Vina terms. Vina terms are calculated using the \
		git repository(https://github.com/HongjianLi/RF-Score) that implements rf-score v3.")
	parser.add_argument("--input_file", "-i", required = True, type = input_check, nargs = "+",
			help = "The file containing the paths to the structure files of the protein and ligand \
			used to calculate the Vina terms.")	
	parser.add_argument("--output_file", "-o", required = True, type = output_check, nargs = "+",
			help = "This file saves the Vina terms of the protein-ligand complex.")	
	
	args = parser.parse_args()		

	if len(args.input_file) != 1:
		raise IOError("Please input only one file containing the path of the structure files.")
	input_file = args.input_file[0]  
	
	if len(args.output_file) != 1 :
		raise IOError("Please input only one file path for save the Vina terms.")
	output_file = args.output_file[0] 
	
	print("===============================================================================")
	print("[Start] '%s' input file is being preprocessing..."%(input_file.strip().split("/")[-1]))
	protein_list, ligand_list = sep_input_file(input_file)

	if len(protein_list) != len(ligand_list):
		raise IOError("The number of proteins and ligands is not correct.")
	complex_count = len(protein_list)
	
	check_file_format(protein_list, ligand_list)
	
	print("\n\t#Start calculation for '%s protein' and '%s ligand'.\n"%(complex_count, complex_count))
	make_Vina_terms(protein_list, ligand_list, output_file)
	print("[Finish] '%s' output file is generated..." %(output_file.strip().split("/")[-1]))
	print("===============================================================================")

'''
Calculate Vina terms.
'''	
def make_Vina_terms(protein_list, ligand_list, output_file):
	
	Vina_terms = dict()
	rf_score_dir = os.path.abspath("./RF-Score")

	for p, l in zip(protein_list, ligand_list):	
		output = os.path.abspath("./tmp.csv")
		
		os.chdir(rf_score_dir)	
		command = "rf-extract " + p + " " + l + " > " + output
		os.system(command)
	
		name, vina_terms = preprocessing_(p, l, read_file(open(output, "r")))
		Vina_terms[name] = vina_terms
	
		os.system("rm " + output)
		
	with open(output_file, "wb") as f:
		pickle.dump(Vina_terms, f)

'''
Extract only 6 Vina terms. 
'''		
def preprocessing_(protein_path, ligand_path, lines):
	
	index = [36, 37, 38, 39, 40, 46]
	
	protein_name = protein_path.strip().split("/")[-1].split(".")[0]
	ligand_name = ligand_path.strip().split("/")[-1].split(".")[0]
	
	name = protein_name + "/" + ligand_name
	
	for line in lines:
		tmp = line.strip().split(",")
		
	return name, np.array(tmp, dtype = np.float32)[index]

'''
Check the file format of the input files.
'''	
def check_file_format(protein_list, ligand_list):
	
	for p, l in zip(protein_list, ligand_list):
		
		protein_file_format = p.strip().split("/")[-1].split(".")[-1]
		ligand_file_format = l.strip().split("/")[-1].split(".")[-1]

		if protein_file_format != "pdbqt" or ligand_file_format != "pdbqt":
			raise IOError("Please check the file format.")	

def sep_input_file(input_file):
	
	lines = read_file(open(input_file,"r"))
	lines = np.array(preprocessing(lines))
	
	return list(lines[:,0]), list(lines[:,1])
	
def preprocessing(lines):
	return [line.strip().split("\t") for line in lines[1:]]
	
def input_check(path):

	path = os.path.abspath(path)
	if not os.path.exists(path):
		raise IOError('%s does not exist.' %path)
	return path
	
def output_check(path):
		
	path = os.path.abspath(path)
	dir_path = os.path.dirname(path)
	if not os.access(dir_path, os.W_OK):
		raise IOError("%s cannot be created."%path) 
	return path

def read_file(file):
	return file.readlines()	
	
if __name__ == "__main__":
	main()		