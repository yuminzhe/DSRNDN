# DSRNDN: A Dilated SE-ResNet and DenseNet Ensemble Model for Protein Contact Map Prediction
The DSRNDN network is composed of 1DResnet, 2D Dilated Se-ResNet and 2D DenseNet. We input the 1D feature and 2D feature of the protein sequence into the DSRNDN network to predict the protein contact map.

## Install

To use DSRNDN you must make sure that your python version is greater than 3.6.
```python
python
>>> import platform
>>> platform.python_version()
'3.6.10'
```

* PyPI  
Directly install the required packages from PyPI.

```bash
pip install tensorflow==1.14.0
pip install tf_learn==0.32

```
We recommend using conda to install hhsuite:
```bash
conda install -c conga-forge -c bioconda hhsuite 
```


## Usage
### Generate multiple sequence alignments

1.Install HHsuite for MSA generation (https://github.com/soedinglab/hh-suite)
In addition to the HHsuite package itself, download the HHsuite-specific sequence database and unzip it into a folder. The sequence database can be downloaded from https://uniclust.mmseqs.com.

2.Install EVcouplings for generating MSAs by jackhmmer 
It is available at https://github.com/debbiemarkslab/EVcouplings . Although the whole package will be installed, only the MSA generation module will be used.The sequence database can be downloaded from https://www.uniprot.org/uniref/.


### Feature extraction

1.Get CCMpred feature
Install CCMpred from https://github.com/soedinglab/CCMpred
Then you can run the following code in your command lines to get CCMpred feature.
```bash
ccmpred [options] input.aln output.mat
```

2.Get ESM-1b feature
Install Evolutionary Scale Modeling from https://github.com/facebookresearch/esm. We used ESM-1b embedding before, and now facebookresearch has launched a new model (ESM-MSA-1).
The use steps are as follows：

It supports PyTorch Hub
```bash
import torch
model, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
```
Then, you can load and use a pretrained model as follows:
```bash
import torch
import esm

# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]
```
3.Get other 1D and 2D features
How to get other 1D and 2D features can be viewed in feature_extraction.py

### How to train
In order to reduce the feature reading time, the extracted features are packaged into pkl format in the form of a dictionary
```bash
with open('../raw_data/feature.pkl','wb') as f:
    pickle.dump(d1,f)
```
Then modify the training set path of config.py
```bash
tf.app.flags.DEFINE_string(
    'train_file', 'pdb_train.pkl',
    'Directory where checkpoints and event logs are written to.')
```
Finally, run the following command line to start training
```bash
python train.py
```

