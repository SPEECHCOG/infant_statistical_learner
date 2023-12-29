# infant_statistical_learner

This repository contains the instructions and scripts for the experiments in:
Khorrami & Räsänen:"Statistical learning is sufficient for early language acquisition: evidence from a modeling study with real-world input statistics", 2023

# Model Source

This project's model is based on the work from the following repositories, please ensure that you give credit to the creators for their contributions to the model:

https://github.com/jasonppy/FaST-VGS-Family


https://github.com/jasonppy/word-discovery


The model (VG-W2V2) and data loading scripts are adopted from https://github.com/jasonppy/word-discovery, whereas the model training and validation scripts are adopted from https://github.com/jasonppy/FaST-VGS-Family, for simultaneouse training of VGS and wave2vec 2.0 piplelines. All changes are commented in the corresponding script lines.


# Model Description

The model is VG-W2V2 and conists of an image procesing pipeline, a speech processing pipeeine, both trained in unsupervised manner, as well as a cross-modal learning pipeline that maps visual and auditory embeddings together and trained in a weakly supervised manner. The image processing pipeline is a DINO model that is a ViT (vision transformer) model and is traiend in unsupervised manner. The pipeline uses the pretrained weights from DINO-small. The speech processing pipeline is a wav2vec 2.0 model, and similar to the original model is trained using a combination of reconstruction (for the masking block) and diversity (for the quantization block) losses. For the speech self-supervised learning task, the speech processing block is randomly initialized. In the visually grounded speech processing pipeline, the audio and visual emdeddings are mapped together through a similarity score to form a shared semantic space. The visually grounded speech (VGS) processing pipeline is trained using a contrastive loss that tries to map similar speech-image pairs closer together is the semantic space and pull away unsimilar pairs. The joint model of speech self-supervised learning and visually grounded speech processsing is trained by combining the audio loss (wav2vec 2.0 pipeline) and the cross-modal contrastive loss (VGS pipeline). 

# Data

# Data subsets

# How to Use


