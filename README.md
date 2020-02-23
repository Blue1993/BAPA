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
The protein structure of PDB format is converted to mol2 format using [UCSF Chimera](https://en.wikipedia.org/wiki/UCSF_Chimera). If you already have the mol2 file of protein, you can skip this step, but we recommend using the file converted with UCSF Chimera.

Convert using `get_mol2.ipynb` notbook. Specify `dir' where the structure file is saved as follows.
```
df
```

### 3. Get Vina terms
Five intermolecular Vina terms and one flexible Vina terms are calculated using the protein and ligand structure in the PDBQT format. The network predicts the binding affinity using six Vina terms and the number of occurrences of each descriptor.

For more details,
```
python prepare.py -h
```

### 4. Get occurrence of descriptors
Calculate the number of occurrences of each descriptor within the given complex using the mol2 structure of the protein and ligand. Distance threshold and the number of descriptors is fixed at 12Å, 2,500, respectively. The Result of `get_descriptor_occurrence.py` script is saved in binary file format. 

```
python get_descriptor_occurrence.py -m 1 -p ./data/complexes/10gs_protein.mol2 -l ./data/complexes/10gs_ligand.mol2 -o ./data/dataset/test_data.pkl
```

```
실행코드
```

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
python get_descriptor_occurrence.py -h
```

# Predict
위의 4가지 step을 Protein-ligand complex의 binding affinity을 예측할 수 있는 trained network가 포함되어 있다. 
```
실행 코드
```

```
실행 결과
```


For more details,
```
python prepare.py -h
```

# Train

For more details,
```
python prepare.py -h
```

# References
