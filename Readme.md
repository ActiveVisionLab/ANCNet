# Release demo

This package contains a demo for submission #689 Correspondence Networks with Adaptive Neighbourhood Consensus. 

The demo first calculates the pck@0.1 score on the PF-PASCAL dataset and then export two sets of visualisations. The first shows the correlation map and key point predictions. The second set illustrates the key point matching results.orrelation map with and key point predictions. The second set illustrates the key point matching results. The visualisation can be found in the folder output.


## Requirements
- Ubuntu 18.04 
- Conda 
- python 3.7
- CUDA 9.0 or newer

## Installation
1. Install CUDA 9.0 as well as either anaconda or miniconda [link](https://docs.conda.io/en/latest/miniconda.html#linux-installers)
2. Create a conda environment: `conda create -n 689release python=3.7`
3. Activate the environment: `conda activate 689release`
4. Run the following commands:
    - `conda install pytorch torchvision cudatoolkit=9.0`
    - `pip install -r requirements.txt`
    - `wget -O ancnet.zip https://www.dropbox.com/s/bjul4f5z7beq3um/ancnet.zip?dl=0 && unzip -q ancnet.zip && rm ancnet.zip`

## Usage
To run example code: `python eval_pf_pascal.py`

## Quick start
After creating a conda environment, you can simply do $sh run.sh. If you encounter any error, please follow installation and usage sections. 
