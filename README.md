# Contextual xLSTM-Based Multimodal Fusion for Conversational Emotion Recognition

Quick Start
Environment setup

# Environment
Python 3.8.13 + Torch 2.4.1 + CUDA 11.8
conda create --name MFCRE python=3.8
conda activate MFCRE
# Hardware: 
single RTX 3090 GPU


Dataset
https://github.com/TaoShi1998/MultiEMO

Please place the dataset as follows:
Dataset/IEMOCAP
Dataset/MELD


# Install dependencies
cd MFCRE
pip install -r requirements.txt


# Run the model
IEMOCAP Dataset
bash Train/TrainMultiEMO_IEMOCAP.sh

MELD Dataset
bash Train/TrainMultiEMO_MELD.sh
