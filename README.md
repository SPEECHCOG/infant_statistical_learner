# infant_statistical_learner

This repository contains the instructions and scripts for the experiments in:
Khorrami & Räsänen:"Statistical learning is sufficient for early language acquisition: evidence from a modeling study with real-world input statistics", 2023

# Model Source

This project's model is based on the work from the following repositories, please ensure that you give credit to the creators for their contributions to the model:

https://github.com/jasonppy/FaST-VGS-Family


https://github.com/jasonppy/word-discovery


Especically, the scripts in "datasets", "models", "steps" as well as run_spokencoco.py are copied from the above repositories. The model (VG-W2V2) and data loading scripts are adopted from https://github.com/jasonppy/word-discovery, whereas the model training and validation scripts are adopted from https://github.com/jasonppy/FaST-VGS-Family in order to have simultaneous training of VGS and wav2vec2.0 pipelines. All changes with respect to the original code are commented in the corresponding script lines.


# Model Description

The model is VG-W2V2 and consists of an image procesing pipeline, a speech processing pipeline, both trained in unsupervised manner, as well as a cross-modal learning pipeline that maps visual and auditory embeddings together and is trained in a weakly supervised manner. The image processing pipeline is a DINO model based on ViT (vision transformer) and is trained in an unsupervised manner. The pipeline uses the pretrained weights from DINO-small. The speech processing pipeline is a wav2vec 2.0 model, and similar to the original model is trained using a combination of reconstruction (for the masking block) and diversity (for the quantization block) losses. For the speech self-supervised learning task, the speech processing block is randomly initialized. In the visually grounded speech processing pipeline, the audio and visual emdeddings are mapped together through a similarity score to form a shared semantic space. The visually grounded speech (VGS) processing pipeline is trained using a contrastive loss that tries to map similar speech-image pairs closer together is the semantic space and pull away unsimilar pairs. The joint model of speech self-supervised learning and visually grounded speech processsing is trained by combining the audio loss (wav2vec 2.0 pipeline) and the cross-modal contrastive loss (VGS pipeline). 

# Data

The data for speech self-supervised learning task is SSL-6M that includes speech captions totalling to 1049 hours of speech. The data is a combination of a subset randomly selected from LibriSpeech (https://www.openslr.org/12) training set (175 892 clips, 602.6 h) and SpokenCOCO (https://groups.csail.mit.edu/sls/downloads/placesaudio/index.cgi) training set (370 121 clips, 446.4 h). 

The data for the joint speech self-supervised training and visually grounded speech processing is VGS-infant and includes subsets that reflect the statistics of the infants' audiovisual experinece for the intervals of 2, 4, and 6 months. The images and audio captions are subsets chosen from MSCOCO images (https://cocodataset.org) paired with SpokenCOCO captions. Please the list of the speech and image files that are utilized in each of the SSL-6M and VGS-infant training scenarios in splits.zip file. 

# How to Use

## Model training and validation

Please follow the source repositories for downloading and saving audio files from SpokenCOCO and LibriSpeech audio files, MSCOCO images as well as json file names for the Karpathy split. The main difference here is that instead of the SpokenCOCO_train_unrolled_karpathy.json file, the model reads json files provided for each audiovisual training subset. The subset1, subset2, and subset3 json files include data for 8 months, 10 months, and 12 months age infant, respectively, whereas the subset0A includes data for 10 months uniform distribution. The tsv files used for training and validating 6 months speech-only model are also included in the splits folder.

To train the model for speech self-supervised learning SSL-6M specify the path to ssl6M_root as well as the output exp path (for saving the trained model) in scripts/ssl.sh and run the script. After the training is completed, you can use best_bundle.pth file as starting point for audiovisual learning pipeline. 

To train the VG-W2V2 model for 2-6 months infants' audiovisual experiments (6-12 months of age), specify the path to the subsets folder (which includes json data files for different splits) and the path to the output exp folder in the corresponding script (s8M.sh, s10M.sh, s10u.sh, and s12M.sh) and run the script. The twd path is the path to best_bundle.pth file from 6 months speech self-supervised training. 

## Audiovisual similarity score matrix (S)

After training the models, run semtest.sh script to obtain S matrix used for COCO-Semtest word meaning score (https://github.com/SPEECHCOG/COCO_Semtest). You need to give the test data paths as well as the path to save the S matrix before running the script. Please find the list of test images and audio files in COCO_Semtest repository. 
