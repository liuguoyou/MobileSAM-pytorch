# MobileSAM-distiller
Reproduction of MobileSAM using pytorch

## Installation

Clone the Repository:

```
git clone https://github.com/ChaoningZhang/MobileSAM.git
cd MobileSAM
pip install -e .
```

Download SA-1B dataset parts from [here](https://segment-anything.com/dataset/index.html) and unzip them, and then pre-process the dataset using:

```
# for example, after downloading sa_000000.tar, we unzipped the file into the file folder 'sa_000000',
# we can run 'python preprocess.py --dataset_dir sa_000000' to pre-process the data to generate features' .npy file
python preprocess.py --dataset_dir <dataset_name>
```
