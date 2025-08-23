📦 Installation Guide for DKGTrack

This document provides step-by-step instructions to set up the environment for DKGTrack.
We recommend using Python 3.8 and conda for environment management.

1. Create a Conda Environment
   
   conda create -n dkgtrack python=3.8 -y
   
   conda activate dkgtrack

2. Install PyTorch
   
   pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

3. Build MultiScaleDeformableAttention

   cd ./models/ops
   
   sh ./make.sh
   
5. Install Other Dependencies
   
   pip install torchtext==0.10.0 transformers==4.6.1 tensorboardX==2.4 protobuf==3.17.3 einops==0.3.0 thinc==8.0.7 timm==0.4.12 opencv-python==4.11.0.86 spicy==0.16.0 pandas==2.0.3 seaborn==0.13.2 pycocotools==2.0.7  https://download.pytorch.org/whl/torch_stable.html

6. Install SpaCy and Models
   
   pip install spacy==3.6.1
   
   pip install thinc==8.1.10 pydantic==1.10.12
   
   python3 -m spacy download en_core_web_sm

7. Reinstall timm without dependencies (to avoid conflicts)
    
    pip install timm==0.4.12 --no-deps
