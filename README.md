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

## Usage
### Generate multiple sequence alignments

1.Install HHsuite for MSA generation (https://github.com/soedinglab/hh-suite).</br>
In addition to the HHsuite package itself, download the HHsuite-specific sequence database and unzip it into a folder. The sequence database can be downloaded from https://uniclust.mmseqs.com.

2.Install EVcouplings for generating MSAs by jackhmmer. </br>
It is available at https://github.com/debbiemarkslab/EVcouplings . The sequence database can be downloaded from https://www.uniprot.org/uniref/.


### Feature extraction

1.Get CCMpred feature</br>
Install CCMpred to generate feature (https://github.com/soedinglab/CCMpred).
Run the following code  to get CCMpred feature.
```bash
ccmpred [options] input.aln output.mat
```
2.Get ESM-1b feature</br>
Install Evolutionary Scale Modeling to generate feature(https://github.com/facebookresearch/esm). <br>
The specific steps are as follows:<br>
</br>
(Ⅰ)Download the pretrained model from PyTorch Hub.
```bash
import torch
model, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
```
(Ⅱ)Then, you can load and use the pretrained model as follows:
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
3.Get other 1D and 2D features</br>
How to get other 1D and 2D features can be viewed in https://github.com/yuminzhe/DSRNDN/blob/master/DSRNDN/feature_extraction.py.<br> 
![predicted contact map of 4xmq](https://github.com/yuminzhe/DSRNDN/blob/master/DSRNDN/pic/feature.jpg)<br>

### How to train
First, we need to package the 1D features and 2D features in the form of a dictionary into pkl format.
```bash
with open('../raw_data/feature.pkl','wb') as f:
    pickle.dump(d1,f)
```
Then we need modify the training path in the https://github.com/yuminzhe/DSRNDN/blob/master/DSRNDN/libs/config/config.py.
```bash
tf.app.flags.DEFINE_string(
    'train_file', 'pdb_train.pkl',
    'Directory where checkpoints and event logs are written to.')
```
Finally, run the following command line to start training
```bash
python train.py
```

###  predicted contact map of 4xmq

![predicted contact map of 4xmq](https://github.com/yuminzhe/DSRNDN/blob/master/DSRNDN/pic/4xmq.png)<br>

