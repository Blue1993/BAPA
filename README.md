# BAPA
BAPA is a neural network for predicting binding affinity of protein-ligand complexes. The network was trained with PDBbind databased and tested with CASF and CSAR "scoring power" benchmark.

# Requirements
python 3.6.8  
Open Babel 2.4.1  
tensorflow 1.12.0  
numpy 1.16.4  
scikit-learn 0.22  
UCSF Chimera  

# Prepare complexes
### 1. Remove water and convert to PDBQT
Water of protein structure is removed and the format of the protein and ligand structure is converted to [PDBQT](http://autodock.scripps.edu/faqs-help/faq/what-is-the-format-of-a-pdbqt-file) using [open babel](http://openbabel.org/wiki/Main_Page). The format of protein structure is [PDB](https://en.wikipedia.org/wiki/Protein_Data_Bank_(file_format)) and the format of ligand structure is [mol2](http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf). 

Input the file containing the path of the structure file of the protein and ligand to be converted. 
```
python convert_to_PDBQT.py -i input_list_PDBQT.txt -o ./data/complexes
```
For more details,
```
python convert_to_PDBQT.py -h
```

### 2. Convert to mol2  
The protein structure of PDB format is converted to mol2 format using [UCSF Chimera](https://en.wikipedia.org/wiki/UCSF_Chimera). If you already have the mol2 file of protein, you can skip this step, but we recommend using the file converted with UCSF Chimera.

To convert a PDB file to mol2 file, use `convert_to_mol2.ipynb` notbook. 

If you wnat to change the `Dir`, change the path in `convert_to_mol2.ipynb` notebook as follows:
```
path = "./data/complexes"
```

### 3. Get Vina terms
Five intermolecular Vina terms and one flexible Vina term are calculated using the protein and ligand structure in the PDBQT format. The Vina terms used in the proposed method are calculated using a [git repository](https://github.com/HongjianLi/RF-Score) that implements rf-score v3. The network predicts the binding affinity using six Vina terms and the number of occurrences of each descriptor.

Input the file containing the path of the structure file of the protein and ligand.
```
python get_Vina_terms.py -i input_list_Vina_terms.txt -o ./data/dataset/Vina_terms.pkl
```
For more details,
```
python get_Vina_terms.py -h
```

### 4. Get occurrence of descriptors
Calculate the number of occurrences of each descriptor within the given complex using the mol2 structure of the protein and ligand. Distance threshold and the number of descriptors is fixed at 12Å, 2,500, respectively. The Result of `get_descriptors_occurrence_count.py` script is saved in binary file format. 

Input the file containing the path of the structure file of the protein and ligand.
```
python get_descriptors_occurrence_count.py -i input_list_count.txt -o ./data/dataset/dataset.pkl
```
For more details,
```
python get_descriptors_occurrence_count.py -h
```

# Predict
When Vina terms and the number of occurence of descriptors are ready, you can predcit the binding affinity using the BAPA.
```
python predict.py -d ./data/dataset/dataset.pkl -v ./data/dataset/Vina_terms.pkl -o ./result/result.csv
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
In order to train the model, label information is required, so the `dataset.pkl` file created for prediction needs to be regenerated as follows.
```
python get_descriptors_occurrence_count.py -i input_list_count_train.txt -o ./data/dataset/dataset.pkl
```
The `input_list_count_train.txt` file should contain the label information in the last tab.

When all preparation is complete, you can train the BAPA model as follows:
```
python training.py -d ./data/dataset/dataset.pkl -v ./data/dataset/Vina_terms.pkl -s ./module.train_network 
```

For more details,
```
python training.py -h
```
