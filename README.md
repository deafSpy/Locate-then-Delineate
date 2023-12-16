# Locate-Then-Delineate: A free-text report guided approach for pneumothorax segmentation in chest radiographs

Created by Samruddhi Shastri*, Naren Akash RJ*, Lokesh Gautham, Jayanthi Sivaswamy

This is an official PyTorch implementation of the [Locate-Then-Delineate: A free-text report guided approach for pneumothorax segmentation in chest radiographs](https://cvit.iiit.ac.in/mip/projects/ptxseg/)

We present a novel solution for accurate segmentation
of pneumothorax from chest radiographs utilizing free-text radiology reports. Our solution employs text-guided attention to leverage the findings in the report to initially produce a low-dimensional region-localization map. These prior region
maps are integrated at multiple scales in an encoder-decoder segmentation framework via dynamic affine feature map transform (DAFT). Extensive experiments on a public dataset CANDID-PTX, show that the integration of free-text reports significantly reduces the false positive predictions, while the
DAFT-based fusion of localization maps improves the positive cases. In terms of DSC, our proposed approach achieves 0.60 and 0.95 for positive and negative cases, respectively, and 0.70 to 0.85 for medium and large pneumothoraces.

## Requirements

Python == 3.7 and install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```
Questions about NumPy version conflict. The NumPy version we use is 1.17.5. We can install bert-embedding first, and install NumPy then.

## Data Preparation
Information on the CANDID-PTX dataset can be found in their paper here: https://pubs.rsna.org/doi/10.1148/ryai.2021210136 and the data can be accessed after signing a data use agreement and an online ethics course: https://pubs.rsna.org/doi/10.1148/ryai.2021210136

### Format Preparation
Then, prepare the datasets in the following format for easy use of the code:
```angular2html
├── SCRATCH_FOLDER_PATH
    ├── candid_ptx_dataset
    │   ├── dicom_files
    │   ├── masks
    │   ├── texts
    │   └── encoded_embeddings
    └── sample
        ├── dicom_files
        ├── masks
        ├── texts
        └── encoded_embeddings
```
'dicom_files' folder contains the chest radiographs

'masks' folder contains the ground truth masks in the naming format <image_name>.jpg

'texts' folder contains the radiology reports in the naming format <image_name>.txt

To generate a 5-fold stratified dataset, run generate_stratified_folds.py script.

## Usage

To train the model, run the following command
```
python3 main.py <path_to_config_file>
```

To generate inference from a trained model, run the following command -
```
python3 inference.py <path_to_config_file>
```
