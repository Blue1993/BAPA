import numpy as np
import argparse, os

def main():
	
	parser = argparse.ArgumentParser(
		description = "This script converts protein structure file in PDB format to PDBQT format and \
						ligand structure file in mol2 format to PDBQT format for Vina terms calculations.\
						This conversion is done using open babel."
		)	
	parser.add_argument("--input_file", "-i", required = True, type = _check, nargs = "+",
			help = "The file that records the path of the protein and ligand to be converted into PDBQT file. \
					Protein file format must be PDB and ligand file format must be mol2.")	
	parser.add_argument("--output_dir", "-o", required = True, type = _check, nargs = "+",
			help = "Directory to save PDBQT files of protein and ligand.")	
	
	args = parser.parse_args()
	
	if len(args.input_file) != 1:
		raise IOError("Please input only one file containing the path of the structure files.")
	input_file = args.input_file[0]
	
	if len(args.output_dir) !=1 :
		raise IOError("Please input only one Dir path for save PDBQT files.")
	output_dir = args.output_dir[0]
	
	print("===============================================================================")
	print("[Start] '%s' input file is preprocessing..."%(input_file.strip().split("/")[-1]))
	
	protein_list, ligand_list = sep_input_file(input_file)
	check_file_format(protein_list, ligand_list)	
	
	if len(protein_list) != len(ligand_list):
		raise IOError("The number of proteins and ligands is not correct.")
		
	complex_count = len(protein_list)
	
	print("\n\t#Convert '%s protein' and '%s ligand' into '%s pdbqt' file.\n"%(complex_count, complex_count, complex_count))
	protein_list = remove_water(protein_list, output_dir)
	make_pdbqt(protein_list, ligand_list, output_dir)
	print("[Finish] Check the '%s' dir..." %output_dir)	
	print("===============================================================================")
	
'''
Convert protein PDB file and ligand mol2 file to PDBQT file with openbabel.
'''
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

'''
Remove water from the input PDB file.
'''		
def remove_water(protein_list, output_dir):	
	
	remove_water_protein_list = list()
	
	for i in protein_list:
		
		output_file = output_dir + "/" + i.split("/")[-1].split(".")[0] + ".pdb"

		lines = [line.strip() for line in open(i,"r")]
		fwrite(lines, open(output_file, "w"))
		
		remove_water_protein_list.append(output_file)
		
	return remove_water_protein_list

'''
Save the PDB file with water removed.
'''
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

'''
Check the file format of the input files.
'''
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
	
	data = list()
	for line in lines[1:]:
		data.append([os.path.abspath(i.strip()) for i in line.strip().split("\t")])
	return data
	
def read_file(file):
	return file.readlines()	

def _check(path):

	path = os.path.abspath(path)
	if not os.path.exists(path):
		raise IOError('%s does not exist.' %path)
	return path
	
if __name__ == "__main__":
	main()	
	