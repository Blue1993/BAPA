import numpy as np
import argparse, os, pickle
from sklearn.metrics.pairwise import euclidean_distances

def main():
	
	parser = argparse.ArgumentParser(
		description = "This script takes in the mol2 files of protein and ligand calculates\
		the number of occuurrences of each descriptor. Distance threshold and the number of descriptors \
		is fixed at 12Å, 2,500, respectively."
	)
	parser.add_argument("--input_file", "-i", required = True, type = input_check, nargs = "+",
			help = "The file containing the path of the structure file of protein and ligand\
			to calculate the number of occurrence of each descriptors.\
			If you want to create a dataset for training, you neet to add an third tab for experimental binding affinity\
			for the complex.")	
	parser.add_argument("--output_file", "-o", required = True, type = output_check, nargs = "+",
			help = "This file is to save the number of occurrences of each descriptors calculated.")	
			
	args = parser.parse_args()	
	
	if len(args.input_file) != 1:
		raise IOError("Please input only one file containing the path of the structure files.")
	input_file = os.path.abspath(args.input_file[0])

	if len(args.output_file) != 1 :
		raise IOError("Please input only one file path for save number of occurrences of each descriptors file.")
	output_file = os.path.abspath(args.output_file[0])
		
	'''  
	First index of input_list is about protein.
	Second index of input_list is about ligand.
	If you add affinity in input file, third index of input_list is about experimental affinity.
	If there is no third tab, the third index of input_list is a list filled with all ones.
	'''
	print("===============================================================================")
	print("[Start] '%s' input file is being preprocessing..."%(input_file.strip().split("/")[-1]))
	
	protein_list, ligand_list, labels_list = sep_input_file(input_file)

	if len(protein_list) != len(ligand_list):
		raise IOError("The number of proteins and ligands is not correct.")
	complex_count = len(protein_list)
	
	check_file_format(protein_list, ligand_list)

	print("\n\t#Start calculation for '%s protein' and '%s ligand'.\n"%(complex_count, complex_count))

	if np.sum(labels_list) == len(labels_list):
		graphs_dict, _ = read_complexes(protein_list, ligand_list, labels_list)

		data = make_data(graphs_dict)

		with open(output_file, "wb") as f:
			pickle.dump(data, f)			
	else:
		graphs_dict, labels = read_complexes(protein_list, ligand_list, labels_list)
		data = make_data(graphs_dict)

		with open(output_file, "wb") as f:
			pickle.dump((data, labels), f)	
	
	print("[Finish] '%s' output file is generated..." %(output_file.strip().split("/")[-1]))
	print("===============================================================================")

def make_data(graphs_dict):	
	
	with open("./module/selected_descriptors_list.pkl","rb") as f:
		selected_descriptors = pickle.load(f)
	
	data = dict()
	
	''' For each complex '''
	for name in graphs_dict.keys():
		whole_descriptors = dict()
		
		for type in graphs_dict[name].keys():
			
			''' 
			one descriptor check  
			e.g. (16, 16):[{(1, 16, 6, '1'): 2, (0, 16, 6, '1'): 1}, ...]	
			'''
			for descriptor in graphs_dict[name][type]:
				if tuple(sorted(descriptor.items())) in whole_descriptors:
					whole_descriptors[tuple(sorted(descriptor.items()))] += 1
				else:
					whole_descriptors[tuple(sorted(descriptor.items()))] = 1	
		
		''' Create a row vector of size 2,500 for each complex. '''
		row_vetor = list()
		for selected_descriptor in selected_descriptors:
			row_vetor.append(whole_descriptors[selected_descriptor]) if selected_descriptor in whole_descriptors else row_vetor.append(0)
				
		data[name] = np.array(row_vetor, dtype = np.float32)	
		
	return data
	
def read_complexes(protein_list, ligand_list, labels_list):
	
	data, labels = dict(), dict()
	
	for idx, (p_file, l_file) in enumerate(zip(protein_list, ligand_list)):
		
		''' 
		ligand atoms type : Store SYMBL of each atom in list.
		e.g. ['O.co2', 'P.3', 'O.co2', 'O.co2', 'C.3', 'N.4', 'C.3', 'C.3', 'C.3', 'C.3', 'C.3', 'C.3', 'C.3']
		ligand atoms type index dict : Store index for nine atom types.
		key : (atom number), values : index list // e.g. {8: [0, 2, 3], 15: [1], 6: [4, 6, 7, 8, 9, 10, 11, 12], 7: [5]}
		ligand atoms coords : The coordinates of each atom.
		ligand bonds : Store each atom's bond in an undirected graph((start, end),(start, end)).
		'''
		ligand_atoms_type, ligand_atoms_type_index_dict, ligand_atoms_coords, ligand_bonds = read_ligand(read_file(open(l_file, "r")))
		
		'''
		protein atoms type : Store SYMBL of each atom in list.
		protein atoms type index dict : Store index for nine atom types.
		protein atoms coords : The coordinates of each atom.
		protein bonds : Store each atom's bond in an undirected graph((start, end),(start, end)).		
		'''
		protein_atoms_type, protein_atoms_type_index_dict, protein_atoms_coords, protein_bonds = read_protein_file(read_file(open(p_file, "r")))

		'''
		Find all cases where the distance between the ligand and the protein atom is less than a certain threshold.
		interaction list -> key : (protein atom type, ligand atom type), 
		value : [[ligand atom index, protein atom index], [ligand atom index, protein atom index], ...]
		'''
		interaction_list = get_interaction(ligand_atoms_type_index_dict, ligand_atoms_coords, protein_atoms_type_index_dict, protein_atoms_coords)
		
		''' find graph '''
		graphs = r_radius_graph(interaction_list, ligand_bonds, ligand_atoms_type, protein_bonds, protein_atoms_type)
		
		protein_name = p_file.split("/")[-1].split(".")[0]
		ligand_name = l_file.split("/")[-1].split(".")[0]
		
		name = protein_name + "/" + ligand_name
		
		data[name] = graphs
		labels[name] = labels_list[idx]
		
	return data, labels
	
def r_radius_graph(interaction_list, ligand_bonds, ligand_atoms_type, protein_bonds, protein_atoms_type):	

	'''
	undirect, atom = [atom1, atom2, ...] 
	bond dict => start : [end1, end2, ...]
	'''
	graphs = dict()
	ligand_bond_start_end, protein_bond_start_end = make_bond_dict(ligand_bonds, protein_bonds)
	
	for i in interaction_list.keys():
		
		sub_interaction_list = interaction_list[i]
		graph_list = list()
		
		''' Extract one protein-ligand interaction(pair) '''
		for val in sub_interaction_list:
			
			sub = dict()
			
			''' find ligand descriptor ''' 
			cur_ligand_atom_index, cur_ligand_atom = val[0], ligand_atoms_type[val[0]]
			
			visited_ligand, step = set(), 0
			visited_ligand.add(cur_ligand_atom_index)
			
			''' Find the one-step neighborhoods of the current atom. '''
			next_ligand_atom_list, visited_ligand = get_next_atom(cur_ligand_atom_index, ligand_bond_start_end, visited_ligand)
			
			for next_ligand_atom_index in next_ligand_atom_list:
				bond = ligand_bonds[(cur_ligand_atom_index, next_ligand_atom_index)]
				next_ligand_atom = ligand_atoms_type[next_ligand_atom_index]
				
				if ((1, cur_ligand_atom, next_ligand_atom, bond)) in sub:
					sub[(1, cur_ligand_atom, next_ligand_atom, bond)] += 1
				else:
					sub[(1, cur_ligand_atom, next_ligand_atom, bond)] = 1
					
			''' find protein descriptor ''' 
			cur_protein_atom_index, cur_protein_atom = val[1], protein_atoms_type[val[1]]
			
			visited_protein, step = set(), 0
			visited_protein.add(cur_protein_atom_index)
			
			''' Find the one-step neighborhoods of the current atom. '''
			next_protein_atom_list, visited_protein = get_next_atom(cur_protein_atom_index, protein_bond_start_end, visited_protein)
			
			for next_protein_atom_index in next_protein_atom_list:
				bond = protein_bonds[(cur_protein_atom_index, next_protein_atom_index)]
				next_protein_atom = protein_atoms_type[next_protein_atom_index]
				
				if ((0, cur_protein_atom, next_protein_atom, bond)) in sub: 
					sub[(0, cur_protein_atom, next_protein_atom, bond)] += 1
				else:
					sub[(0, cur_protein_atom, next_protein_atom, bond)] = 1				
			
			graph_list.append(sub)
		
		graphs[i] = graph_list		
	
	return graphs

'''
Find one-step neighborhoods of the current atom.
'''	
def get_next_atom(current_atom, bond_dict, visited_atoms):

	next_atoms_list = bond_dict[current_atom]
	next_atoms_list = list(set(next_atoms_list) -  visited_atoms)	
	
	if len(next_atoms_list) == 0:
		return [], visited_atoms
	else:
		for i in next_atoms_list:
			visited_atoms.add(i)
			
		return next_atoms_list, visited_atoms

'''
Stores bonds between atoms in a dictionary.
'''	
def make_bond_dict(ligand_bonds, protein_bonds):
	
	ligand_start_end, protein_start_end = dict(), dict()
	
	for i in ligand_bonds.keys():
		start = i[0]
		end = i[1]
		
		if start in ligand_start_end:
			ligand_start_end[start].append(end)
			
		else:	
			ligand_start_end[start] = [end]
			
	for i in protein_bonds.keys():	
		start = i[0]
		end = i[1]
	
		if start in protein_start_end:
			protein_start_end[start].append(end)
		else:
			protein_start_end[start] = [end]
			
	return ligand_start_end, protein_start_end

'''
The protein and ligand atom pairs that exist with a certain distance are stored in
36 different types.
'''	
def get_interaction(ligand_atoms_type_index_dict, ligand_atoms_coords, protein_atoms_type_index_dict, protein_atoms_coords):
	
	interaction_list = make_interaction_dict()	
	
	for i in protein_atoms_type_index_dict.keys():
		
		protein_atom_list = protein_atoms_type_index_dict[i]
		protein_atom_list_coords = protein_atoms_coords[protein_atom_list]
		
		for j in ligand_atoms_type_index_dict.keys():
			
			ligand_atom_list = ligand_atoms_type_index_dict[j]
			ligand_atom_list_coords = ligand_atoms_coords[ligand_atom_list]
			
			interaction_ligand_atoms_index, interaction_protein_atoms_index, interaction_distance_matrix = cal_distance(ligand_atom_list_coords, protein_atom_list_coords) 

			for ligand_, protein_ in zip(interaction_ligand_atoms_index, interaction_protein_atoms_index):
				interaction_list[(i,j)].append((ligand_atom_list[ligand_], protein_atom_list[protein_]))
			
	return interaction_list

'''
Find protein and ligand atom pairs within a certain distance.
'''	
def cal_distance(ligand_coords, protein_coords):
	
	threshold = 12.
	distance = euclidean_distances(ligand_coords, protein_coords)
	rows, cols = np.where(distance <= threshold)
		
	return rows, cols, distance

'''
Initialize a dictionary to store 36 interactions (protein-ligand pairs).
'''	
def make_interaction_dict():
	
	interaction_list = dict()
	
	protein_atoms_list = [6, 7, 8, 16]
	ligand_atoms_list = [6, 7, 8, 9, 15, 16, 17, 35, 53]
	
	for i in protein_atoms_list:
		for j in ligand_atoms_list:
			interaction_list[(i,j)] = list()
			
	return interaction_list
			
def read_protein_file(lines):
	
	atoms_reference_dict = get_atoms_reference()
	residue_reference_set = get_residue_reference()
	residue_list, atoms_dict, protein_atoms_type, protein_atoms_type_index_dict, protein_atoms_coords, protein_bonds = list(), dict(), list(), dict(), list(), dict()
	flag, index, atom_count = 0, 0, 0	

	while index < len(lines):
		
		''' Extract information about the atom '''
		line = lines[index].strip()	
	
		if flag == 1:
			if "@<TRIPOS>BOND" not in line:
				line_list = remove_sep(line.split(" "))
				
				''' Extract SYMBL type of atom '''
				atoms_number = int(line_list[0])
				atom_type = line_list[5]
				residue_name = line_list[7].strip()[:3]	
				
				''' 
				It works if it is not H(hydrogen).
				Save the coordinate information and atom type, and redefine the index.
				'''	
				if atom_type != "H" and residue_name in residue_reference_set and atom_type in atoms_reference_dict:	

					protein_atoms_coords.append([line_list[2], line_list[3], line_list[4]])
					protein_atoms_type.append(atoms_reference_dict[atom_type])
					
					if atoms_reference_dict[atom_type] in protein_atoms_type_index_dict:
						protein_atoms_type_index_dict[atoms_reference_dict[atom_type]].append(atom_count)
						
					else:	
						protein_atoms_type_index_dict[atoms_reference_dict[atom_type]] = [atom_count]
						
					atoms_dict[atoms_number] = atom_count
					atom_count += 1				
	
		elif flag == 2:
			''' Extract informaction about bond. '''
			if "@<TRIPOS>SUBSTRUCTURE" not in line:
				line_list = remove_sep(line.split(" "))	

				start_atom = int(line_list[1]) 
				end_atom = int(line_list[2])
				bond_type = line_list[3]
			
				if start_atom in atoms_dict and end_atom in atoms_dict:
					protein_bonds[(atoms_dict[start_atom], atoms_dict[end_atom])] = bond_type
					protein_bonds[(atoms_dict[end_atom], atoms_dict[start_atom])] = bond_type	
	
		if "@<TRIPOS>ATOM" in line:
			flag = 1
		elif "@<TRIPOS>BOND" in line:
			flag = 2
		elif "@<TRIPOS>SUBSTRUCTURE" in line:
			flag = 3
		
		index += 1
		
	return protein_atoms_type, protein_atoms_type_index_dict, np.array(protein_atoms_coords, dtype = np.float32), protein_bonds		

def read_ligand(lines):
	
	atoms_reference_dict = get_atoms_reference()
	atoms_dict, ligand_atoms_type, ligand_atoms_type_index_dict, ligand_atoms_coords, ligand_bonds = dict(), list(), dict(), list(), dict()
	flag, index, atom_count = 0, 0, 0
	
	while index < len(lines):
		
		line = lines[index].strip()
		''' Extract information about the atom '''
		if flag == 1:
			if "@<TRIPOS>BOND" not in line:
				line_list = remove_sep(line.split(" "))

				''' Extract SYMBL type of atom '''
				atoms_number = int(line_list[0])				
				atom_type = line_list[5]
				
				''' 
				It works if it is not H(hydrogen).
				Save the coordinate information and atom type, and redefine the index.
				'''				
				if atom_type != "H" and atom_type in atoms_reference_dict:

					ligand_atoms_coords.append([line_list[2], line_list[3], line_list[4]])
					ligand_atoms_type.append(atoms_reference_dict[atom_type])
					
					if atoms_reference_dict[atom_type] in ligand_atoms_type_index_dict:
						ligand_atoms_type_index_dict[atoms_reference_dict[atom_type]].append(atom_count)
					else:
						ligand_atoms_type_index_dict[atoms_reference_dict[atom_type]] = [atom_count]
						
					atoms_dict[atoms_number] = atom_count
					atom_count += 1
				
		elif flag == 2:
			''' Extract informaction about bond. '''
			if "@<TRIPOS>SUBSTRUCTURE" not in line:		
				line_list = remove_sep(line.split(" "))		
		
				start_atom = int(line_list[1]) 
				end_atom = int(line_list[2])
				bond_type = line_list[3]		
		
				if start_atom in atoms_dict and end_atom in atoms_dict:
					ligand_bonds[(atoms_dict[start_atom], atoms_dict[end_atom])] = bond_type
					ligand_bonds[(atoms_dict[end_atom], atoms_dict[start_atom])] = bond_type		
		
		if "@<TRIPOS>ATOM" in line:
			flag = 1
		elif "@<TRIPOS>BOND" in line:
			flag = 2
		elif "@<TRIPOS>SUBSTRUCTURE" in line:
			flag = 3		
		
		index += 1
	
	return ligand_atoms_type, ligand_atoms_type_index_dict, np.array(ligand_atoms_coords, dtype = np.float32), ligand_bonds

def get_residue_reference():
	return {'LEU', 'MET', 'ILE', 'GLU', 'CYS', 'GLY', 'PHE', 'ASN', 'GLN', 'LYS', 'TYR', 'ARG', 'THR', 'PRO', 'VAL', 'ASP', 'ALA', 'TRP', 'HIS', 'SER'}		

def get_atoms_reference():
	return {'N.3':7, 'O.spc':8, 'O.t3p':8, 'C.3':6, 'O.co2':8, 'I':53, 'N.pl3':7, 'S.2':16, 'O.3':8, 'O.2':8, 'F':9, 'C.cat':6, 'P.3':15, 'C.2':6, 'C.ar':6, 'N.1':7, 'N.ar':7, 'Br':35, 'C.1':6, 'S.o2':16, 'Cl':17, 'N.4':7, 'S.3':16, 'S.o':16, 'N.am':7, 'N.2':7}	

def remove_sep(line_list):
	return [element for element in line_list if element != ""]

'''
Check the file format of the input files.
'''	
def check_file_format(protein_list, ligand_list):
	
	for p, l in zip(protein_list, ligand_list):
		
		protein_file_format = p.split("/")[-1].split(".")[-1]
		ligand_file_format = l.split("/")[-1].split(".")[-1]
		
		if protein_file_format != "mol2" or ligand_file_format != "mol2":
			raise IOError("Please check the file format.")

def sep_input_file(input_file):
	
	lines = preprocessing(read_file(open(input_file,"r")))
	
	if lines.shape[1] == 2:
		return list(lines[:,0]), list(lines[:,1]), list(np.ones(len(lines)))
	elif lines.shape[1] == 3:
		return list(lines[:,0]), list(lines[:,1]), list(np.array(lines[:,2], dtype = np.float32))

def preprocessing(lines):

	data = list()
	for line in lines[1:]:
		data.append([i.strip() for i in line.split("\t")])
	return np.array(data)
	
def read_file(file):
	return file.readlines()	

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