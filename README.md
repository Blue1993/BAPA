# BAPA
BAPA is a neural network for predicting binding affinity of protein-ligand complexes. The network was trained with PDBbind databased and tested with CASF and CSAR "scoring power" benchmark.

# Requrements

# Prepare complexes
### 1. (optionally) Remove water and convert to PDBQT
Water of protein structure is removed and the format of the protein and ligand structure is converted to [PDBQT](http://autodock.scripps.edu/faqs-help/faq/what-is-the-format-of-a-pdbqt-file) using [open babel](http://openbabel.org/wiki/Main_Page). The format of protein structure is [PDB](https://en.wikipedia.org/wiki/Protein_Data_Bank_(file_format)) and the format of ligand structure is [mol2](http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf). If you already have a PDBQT file of protein and ligand with water removed, you can skip this step.  

You can convert the PDB file of protein and the mol2 file of ligand into individual PDBQT files in two ways.

Input the structure file of the protein and ligand to be converted. 
```
python convert_to_PDBQT.py -m 0 -p ./data/complexes/10gs_protein.pdb -l ./data/complexes/10gs_ligand.mol2 
```
Or, input the file containing the path of the structure file of the protein and ligand to be converted.  
```
python convert_to_PDBQT.py -m 1 -f input_list_PDBQT.txt
```
For more details,
```
python convert_to_PDBQT.py -h
```

### 2. (optionally) Convert to mol2  
The protein structure of PDB format is converted to mol2 format using [UCSF Chimera](https://en.wikipedia.org/wiki/UCSF_Chimera). If you already have the mol2 file of protein, you can skip this step, but we recommend using the file converted with UCSF Chimera.

To convert a PDB file to mol2 file, use `convert_to_mol2.ipynb` notbook. 

If you wnat to change the `Dir`, change the path in `convert_to_mol2.ipynb` notebook as follows:
```
path = "./data/complexes"
```

### 3. Get Vina terms
Five intermolecular Vina terms and one flexible Vina term are calculated using the protein and ligand structure in the PDBQT format. The Vina terms used in the proposed method are calculated using a [git repository](https://github.com/HongjianLi/RF-Score) that implements rf-score v3. The network predicts the binding affinity using six Vina terms and the number of occurrences of each descriptor.

Input the structure file of the protein and ligand.
```
python get_Vina_terms.py - m 0 -p ./data/complexes/10gs_protein.pdbqt -l ./data/complexes/10gs_ligand.pdbqt -o ./data/dataset/test_Vina_terms.pkl
```
Or, input the file containing the path of the structure file of the protein and ligand.
```
python get_Vina_terms.py - m 1 -f ./input_list_Vina.txt -o ./data/dataset/test_Vina_terms.pkl
```
For more details,
```
python get_Vina_terms.py -h
```

### 4. Get occurrence of descriptors
Calculate the number of occurrences of each descriptor within the given complex using the mol2 structure of the protein and ligand. Distance threshold and the number of descriptors is fixed at 12Å, 2,500, respectively. The Result of `get_descriptors_occurrence_count.py` script is saved in binary file format. 

Input the structure file of the protein and ligand.
```
python get_descriptors_occurrence_count.py -m 1 -p ./data/complexes/10gs_protein.mol2 -l ./data/complexes/10gs_ligand.mol2 -o ./data/dataset/test_data.pkl
```
Or, input the file containing the path of the structure file of the protein and ligand.
```
python get_descriptors_occurrence_count.py -m 1 -f input_list_count.txt -o ./data/dataset/test_data.pkl
```
For more details,
```
python get_descriptors_occurrence_count.py -h
```

# Predict
When Vina terms and the number of occurence of descriptors are ready, you can predcit the binding affinity using the BAPA.
```
python predict.py -d ./data/dataset/test_data.pkl -t ./data/dataset/test_Vina.pkl -o ./result/result.csv
```
The result of `predict.py` consists of the following forms:
```
name,prediction
protein1/ligand1,binding affinity
protein2/ligand2,binding affinity
protein3/ligand3,binding affinity
```
For more details,
```
python predict.py -h
```

# Train

```
python training.py -d ./ -t ./ -o ./
```

```
python training.py -d ./ -t ./ -o ./ -v ./
```

For more details,
```
python training.py -h
```
