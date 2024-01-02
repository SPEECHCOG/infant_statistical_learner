# # This file is mostly copied from https://github.com/jasonppy/FaST-VGS-Family/blob/master/datasets/libri_dataset.py
# All changes to the otiginal file are commented as "kh"

# mostly borrowed from fairseq/fairseq/data/audio/hubert_dataset.py
import random
import numpy as np
import os
import torch
import torch.nn.functional

import soundfile as sf
from torch.utils.data import Dataset
import csv
import logging
logger = logging.getLogger(__name__)


def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        names = [row[0] for row in reader]
    return names

class LibriDataset(Dataset):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--libri_fn_root", type=str, default="../../../../datavf/ssl_root/", help="root for tsv files")
        parser.add_argument("--libri_max_seq_len", type=float, default=15.6)
        parser.add_argument("--libri_val_bzs", type=int, default=64)
        parser.add_argument("--sample_rate", type=int, default=16000)
        parser.add_argument("--feature_rate", type=int, default=50, help="50")
        parser.add_argument("--label_rate", type = int, default=100, help="the number of labels per second of audio. 100 if mfcc, 50 is MAE features")
        parser.add_argument("--feature_dim", type=int,
            default=100,
            help = "dim feature input to the transformer, if use wav, this arg is omited, else if use spectrogram/fbank/mfcc, the default is 80"
        )
        parser.add_argument("--deltas", action="store_true",
            default=True,
            help = "whether or not add delta and delta-delta to the feature, only effective for spectrogram/fbank/mfcc"
        )
        parser.add_argument("--feature_type", type=str, default="wav", help="choose from wav/spectrogram/fbank/mfcc")
        parser.add_argument("--max_keep_sample_size", type=int, default=16000*80)
        parser.add_argument("--min_keep_sample_size", type=int, default=32000)
    def __init__(self, args, split):
        self.args = args
        self.split = split

        if "train" in split:
            manifest_path = os.path.join(self.args.libri_fn_root, "train.tsv")
        elif "val" in split or "valid" in split or "dev" in split:
            manifest_path = os.path.join(self.args.libri_fn_root, "valid.tsv")
        
        self.audio_root= self.args.data_root # "path/to/data/folder/"
        self.audio_names = read_tsv(
            manifest_path#, self.args.max_keep_sample_size, self.args.min_keep_sample_size
        )

    def __len__(self):
        return len(self.audio_names)

    def calculate_batch_size(self, num_steps):
        print('.........here is calculating .............')
        print('........ len self ........................')
        print(len(self))
        print('........ num steps  ........................')
        print(num_steps)
        print('........ calculated bs ........................')
        #bs = int(np.ceil(len(self)/num_steps))
        # KH: I removed ceil since then it ends iterating LS sooner than SC and produces error
        bs = int(len(self)/num_steps)
        print (bs)
        return bs

    def _LoadAudioLabel(self, fn, label_key):
        #Kh: "label_key" is file name without ".flac" but it is not used here
        x, sr = sf.read(fn, dtype = 'float32')
        assert sr == 16000
        length_orig = len(x)
        if length_orig > 16000 * self.args.libri_max_seq_len: 
            audio_length = int(16000 * self.args.libri_max_seq_len)
            
            # Kh: this part is diferent from COCO audio load
            # in COCO for both train and val we have 
            # x = x[:audio_length]
            
            # here selects a random initial time
            # ..............................................
            if "train" in self.split:
                start_max = length_orig - audio_length
                start = random.choice(range(start_max))
                x_temp = x[start:(start+audio_length)]
                if np.linalg.norm(x_temp) != 0:
                    x = x_temp
                else:
                    x = x[:audio_length]
            else:
                x = x[:audio_length]
            # Kh .......................................... 
            # here starts always from time zero
            x = x[:audio_length]
            # .. ..........................................
            
            x_norm = (x - np.mean(x)) / np.std(x) # normalize per instance
            x = torch.FloatTensor(x_norm)
        else:
            audio_length = length_orig
            new_x = torch.zeros(int(16000 * self.args.libri_max_seq_len))
            x_norm = (x - np.mean(x)) / np.std(x) # normalize per instance
            new_x[:audio_length] = torch.FloatTensor(x_norm) 
            x = new_x
        return x, audio_length

    def __getitem__(self, index):
        fn = os.path.join(self.audio_root, self.audio_names[index])
        label_key = "/".join(fn.split("/")[-4:]).split(".")[0]
        wav, wav_len = self._LoadAudioLabel(fn, label_key)
        return wav, wav_len, label_key


    def collate(self, batch):
        vals = list(zip(*batch))
        collated = {}
        collated['audio'] = torch.nn.utils.rnn.pad_sequence(vals[0], batch_first=True)
        collated['audio_length'] = torch.LongTensor(vals[1])
        collated['id'] = vals[2]
        collated['audio_attention_mask'] = torch.arange(len(collated['audio'][0])).unsqueeze(0) >= collated['audio_length'].unsqueeze(1)
        return collated
