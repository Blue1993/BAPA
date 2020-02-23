import numpy as np
import argparse, os

def main():
	
	parser = argparse.ArgumentParser(
		description = "This script converts protein structure file in PDB format and \
						ligand structure file in mol2 format to PDBQT format for Vina terms calculations.\
						This conversion is done using open babel"
		)	
	
	parser.add_argument("--mode", "-m", required = True, type = int,
			help = "Determine how to input the structure files of the protein and ligand.\
			0 means each structure file, and 1 means the file in which the path of each structure file is recorded."
	)
	parser.add_argument("--protein", "-p", type = _check, nargs = "+",
			help = "The protein file that you want to convert to the PDBQT file. File format must be PDB.")
	parser.add_argument("--ligand", "-l", type = _check, nargs = "+",
			help = "The ligand file that you want to convert to the PDBQT file. File format must be mol2.")	
	parser.add_argument("--file", "-f", type = _check, 
			help = "The file that records the path of the protein and ligand to be converted into PDBQT file. \
					Protein file format must be PDB and ligand file format must be mol2.")	
	parser.add_argument("--output_dir", "-o", type = _check, nargs = "+",
			help = "Directory to store PDBQT files of protein and ligand.")	
			
	args = parser.parse_args()
	
	if args.mode == 0:
		if args.protein == None or args.ligand == None:
			raise IOError("There are no protein or ligand structure file.")
		elif len(args.protein) != len(args.ligand):
			raise IOError("The number of protein and ligand files does not match.")	
		
		protein_list, ligand_list = args.protein, args.ligand
		
	elif args.mode == 1:
		if args.file == None:
			raise IOError("There is no input file.")
		
		protein_list, ligand_list = sep_input_file(args.file)
		
	else:
		raise IOError("Incorrect number for mode option.")

	check_file_format(protein_list, ligand_list)	
	
	if args.output_dir == None:
		args.output_dir = os.path.abspath("./data/complexes")
	
	protein_list = remove_water(protein_list, args.output_dir)

	make_pdbqt(protein_list, ligand_list, args.output_dir)

def make_pdbqt(protein_list, ligand_list, output_dir):
	
	for p, l in zip(protein_list, ligand_list):
		
		print("> File %s is being converted to PDBQT..."%p)
		output_protein = output_dir + "/" + p.split("/")[-1].split(".")[0][:-3] + ".pdbqt"
		command = "babel -ipdb " + p + " -opdbqt " + output_protein
		os.system(command)
		
		print("> File %s is being converted to PDBQT...\n"%l)
		output_ligand = output_dir + "/" + l.split("/")[-1].split(".")[0] + ".pdbqt"
		command = "babel -imol2 " + l + " -opdbqt " + output_ligand
		os.system(command)
		
		command = "rm " + p
		os.system(command)
		
def remove_water(protein_list, output_dir):	
	
	remove_water_protein_list = list()
	
	for i in protein_list:
		
		output_file = output_dir + "/" + i.split("/")[-1].split(".")[0] + "_re.pdb"

		lines = [line.strip() for line in open(i,"r")]
		fwrite(lines, open(output_file, "w"))
		
		remove_water_protein_list.append(output_file)
		
	return remove_water_protein_list

def fwrite(lines, fw):
	
	flag = 0
	for line in lines:
		if "HETATM" in line:
			line_list = remove_sep(line.split(" "))
			residue_name = line_list[3].strip()
			if "HOH" in residue_name:
				flag = 1
				
		if flag == 0:		
			fw.write("%s\n"%line)
		flag = 0
		
	fw.close()
	
def remove_sep(line_list):
	return [element for element in line_list if element != ""]
	
def check_file_format(protein_list, ligand_list):
	
	for p, l in zip(protein_list, ligand_list):
		
		protein_file_format = p.split("/")[-1].split(".")[-1]
		ligand_file_format = l.split("/")[-1].split(".")[-1]
		
		if protein_file_format != "pdb" or ligand_file_format != "mol2":
			raise IOError("Please check the file format.")
		
def sep_input_file(input_file):
	
	lines = read_file(open(input_file,"r"))
	lines = np.array(preprocessing(lines))
	
	return list(lines[:,0]), list(lines[:,1])
	
def preprocessing(lines):
	return [line.split("\t") for line in lines]
	
def read_file(file):
	return file.readlines()	

def _check(path):

	path = os.path.abspath(path)
	if not os.path.exists(path):
		raise IOError('%s does not exist.' %path)
	return path
	
if __name__ == "__main__":
	main()	
	