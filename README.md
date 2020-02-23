# BAPA
BAPA is a neural network for predicting binding affinity of protein-ligand complexes. The network was trained with PDBbind databased and tested with CASF and CSAR "scoring power" benchmark.

# Requrements

# Prepare complexes

### 1. (optionally) Remove water and convert to PDBQT
Water of protein structure is removed and the format of the protein and ligand structure is converted to [PDBQT](http://autodock.scripps.edu/faqs-help/faq/what-is-the-format-of-a-pdbqt-file) using [open babel](http://openbabel.org/wiki/Main_Page). The format of protein structure is [PDB](https://en.wikipedia.org/wiki/Protein_Data_Bank_(file_format)) and the format of ligand structure is [mol2](http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf). If you already have a PDBQT file of protein and ligand with water removed, you can skip this step.  

You can convert the PDB file of protein and the mol2 file of ligand into individual PDBQT files in 2 ways with the -m option:  
1) Enter the structure file of the protein and ligand to be converted as arguments. 
```
python prepare.py -m 0 -p ./data/complexes/10gs_protein.pdb -l ./data/complexes/10gs_ligand.mol2 
```
2) Enter as argument the file containing the path of the structure file of the protein and ligand to be converted.  
```
python prepare.py -m 1 -f input_file_list.txt
```
For more details,
```
python prepare.py -h
```

### 2. (optionally)

### 3. 

### 4. 

# Predict

# Train

# References
