# BAPA
BAPA is a neural network for predicting binding affinity of protein-ligand complexes.  
The network was trained with PDBbind databased and tested with CASF and CSAR "scoring power" benchmark.

# Requrements

# Prepare complexes

### 1. (optionally) Remove water and convert to PDBQT
In this step, water of protein structure and ligand structure is removed and converted into each [PDBQT](http://autodock.scripps.edu/faqs-help/faq/what-is-the-format-of-a-pdbqt-file) file using [open babel](http://openbabel.org/wiki/Main_Page).  
The format of protein structure is [PDB](https://en.wikipedia.org/wiki/Protein_Data_Bank_(file_format)) and the format of ligand structure is [mol2](http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf).  
If you already have a PDBQT file of protein and ligand with water removed, you can skip this step.  

You can convert the PDB file of protein and the mol2 file of ligand into individual PDBQT files in 2 ways with the -m option:
1. 변환하려는 protein과 ligand의 structure 파일을 다음과 같이 인자의 형태로 넘겨준다.  
'''
python prepare.py -m 0 -p ./data/complexes/10gs_protein.pdb -l ./data/complexes/10gs_ligand.mol2 
'''

2. 


### 2. (optionally)

### 3. 

### 4. 

# Predict

# Train

# References
