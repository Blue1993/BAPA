# BAPA
BAPA is a neural network for predicting binding affinity of protein-ligand complexes. The network was trained with PDBbind databased and tested with CASF and CSAR "scoring power" benchmark.

# Requrements

# Prepare complexes

### 1. (optionally) Remove water and convert to PDBQT
Water of protein structure is removed and the format of the protein and ligand structure is converted to [PDBQT](http://autodock.scripps.edu/faqs-help/faq/what-is-the-format-of-a-pdbqt-file) using [open babel](http://openbabel.org/wiki/Main_Page). The format of protein structure is [PDB](https://en.wikipedia.org/wiki/Protein_Data_Bank_(file_format)) and the format of ligand structure is [mol2](http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf). If you already have a PDBQT file of protein and ligand with water removed, you can skip this step.  

You can convert the PDB file of protein and the mol2 file of ligand into individual PDBQT files in two ways.


Enter the structure file of the protein and ligand to be converted as arguments. 
```
python prepare.py -m 0 -p ./data/complexes/10gs_protein.pdb -l ./data/complexes/10gs_ligand.mol2 
```
Or, enter as argument the file containing the path of the structure file of the protein and ligand to be converted.  
```
python prepare.py -m 1 -f input_file_list.txt
```
For more details,
```
python prepare.py -h
```

### 2. (optionally) Convert to mol2  
BAPA는 mol2 file formate의 protein structure을 이용하기 때문에 PDB formate의 파일을 chimera을 이용하여 변환한다. 만약 당신이 이미 protein의 mol2 file을 가지고 있다면 이 과정을 skip해도 되지만 chimera을 이용하여 변환된 파일을 사용하는 것을 추천한다.

변환은 `get_mol2.ipynb` notebook을 이용하면되고 structure 파일이 저장되어 있는 dir을 다음과 같이 path로 잡아주면된다.
```
```

### 3. 

### 4. 

# Predict

# Train

# References
