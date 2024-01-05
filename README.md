# infant_statistical_learner

This repository includes the instructions and scripts for the experiments outlined in the paper by Khorrami & Räsänen titled 'Statistical learning is sufficient for early language acquisition: evidence from a modeling study with real-world input statistics.' 

# Model Source

This project's model is built upon the contributions from the following repositories, and it is essential to acknowledge and credit the creators for their valuable work:

    https://github.com/jasonppy/FaST-VGS-Family
and

    https://github.com/jasonppy/word-discovery

Specifically, the scripts located in the 'datasets,' 'models,' 'steps,' and the 'run_spokencoco.py' file have been derived from the aforementioned repositories. The model (VG-W2V2) and data loading scripts are adopted from the word-discovery, while the model training and validation scripts are adopted from the FaST-VGS-Family. This integration allows for simultaneous training of VGS and Wav2Vec 2.0 pipelines. Any modifications made to the original code are diligently documented with comments in the corresponding script lines.


# Model Description

The model, denoted as VG-W2V2, is a complex system consisting of three main pipelines: an image processing pipeline, a speech processing pipeline, and a cross-modal learning pipeline. Each pipeline is trained with specific methodologies to achieve its respective objectives.

The image processing pipeline utilizes a DINO model based on ViT (Vision Transformer), trained in an unsupervised manner. It incorporates pretrained weights from DINO-small. The speech processing pipeline employs a Wav2Vec 2.0 model. Similar to the original model, it undergoes training with a combination of reconstruction (for the masking block) and diversity (for the quantization block) losses. Notably, the speech processing block is randomly initialized for the speech self-supervised learning task.

In the visually grounded speech processing pipeline, audio and visual embeddings are mapped together through a similarity score, creating a shared semantic space. This process results in the formation of the Visually Grounded Speech (VGS) processing pipeline. The VGS pipeline is trained using a contrastive loss, aiming to bring similar speech-image pairs closer together in the semantic space while pushing apart dissimilar pairs.

The joint model, incorporating both speech self-supervised learning and visually grounded speech processing, is trained by combining the audio loss from the Wav2Vec 2.0 pipeline and the cross-modal contrastive loss from the VGS pipeline.

# Data

The dataset utilized for the speech self-supervised learning task is SSL-6M, encompassing speech captions totaling 1049 hours of speech. This dataset is a combination of a randomly selected subset from the LibriSpeech training set (175,892 clips, 602.6 hours) available at https://www.openslr.org/12, and the SpokenCOCO training set (370,121 clips, 446.4 hours) accessible at https://groups.csail.mit.edu/sls/downloads/placesaudio/index.cgi.

For the joint training of speech self-supervised learning and visually grounded speech processing, the dataset used is VGS-infant. This dataset includes subsets that mirror the statistical patterns in infants' audiovisual experiences at 2, 4, and 6 months intervals. The images and audio captions within VGS-infant are subsets selected from MSCOCO images (https://cocodataset.org) paired with SpokenCOCO captions. The list of speech and image files employed in both SSL-6M and VGS-infant training scenarios is available in the 'splits.zip' file.

# How to Use

## Model training and validation

Please refer to the source repositories for the downloading and storage of audio files from SpokenCOCO and LibriSpeech, MSCOCO images, and JSON file names for the Karpathy split. Notably, in this work, the model reads JSON files specific to each audiovisual training subset instead of utilizing the 'SpokenCOCO_train_unrolled_karpathy.json' file.

The JSON files for 'subset1,' 'subset2,' and 'subset3' include data corresponding to 8 months, 10 months, and 12 months age infants, respectively. Additionally, 'subset0A' contains data for a 10-month uniform distribution. The 'tsv' files used for training and validating the 6-month speech-only model are included in the 'splits' folder.

To train the model for speech self-supervised learning (SSL-6M), specify the path to 'ssl6M_root' and the output experiment path (for saving the trained model) in the 'scripts/ssl.sh.' Execute the script, and after completion, utilize the 'best_bundle.pth' file as a starting point for the audiovisual learning pipeline.

For training the VG-W2V2 model for 2-6 months infants' audiovisual experiments (6-12 months of age), specify the path to the 'subsets' folder (which includes JSON data files for different splits) and the path to the output experiment folder in the corresponding script ('s8M.sh,' 's10M.sh,' 's10u.sh,' and 's12M.sh'). Execute the script, ensuring the 'twd' path points to the 'best_bundle.pth' file from the 6-month speech self-supervised training.

## Audiovisual similarity score matrix (S)

After completing the model training, execute the 'semtest.sh' script to generate the S matrix utilized for the COCO-Semtest word meaning score, as outlined in the repository (https://github.com/SPEECHCOG/COCO_Semtest). Before running the script, ensure you provide the paths for the test data and specify the location to save the S matrix. The repository for COCO_Semtest contains a comprehensive list of test images and audio files for reference.
