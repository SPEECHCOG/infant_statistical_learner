import json
import random
import numpy as np
import os
import torch
import torch.nn.functional
import random
import soundfile as sf
from torch.utils.data import Dataset
import h5py
import pickle
import logging
logger = logging.getLogger(__name__)

from PIL import Image
import torchvision.transforms as transforms


# data_root = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp"
# audio_dataset_json_file = os.path.join(data_root, "SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json")
# with open(audio_dataset_json_file, 'r') as fp:
#     data_json = json.load(fp)
# data = data_json['data']
        
# train_img_dataset_h5py_file = os.path.join(data_root, "coco_img_feat/SpokenCOCO_train_imgfeat.hdf5")
# train_imgid2index_file = os.path.join(data_root, "SpokenCOCO/SpokenCOCO_train_imgid2idex.json")
# train_imgid2ordered_indices_file = os.path.join(data_root, "SpokenCOCO/SpokenCOCO_train_imgid2ordered_indices.pkl")
    
# train_img_data = h5py.File(train_img_dataset_h5py_file, 'r')
# with open(train_imgid2index_file, 'r') as fp:
#     train_img_id2index = json.load(fp)    
# with open(train_imgid2ordered_indices_file, 'rb') as f:
#     train_img_id2ordered_indices = pickle.load(f)
    

class ImageCaptionDataset(Dataset):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--data_root", type=str, default="../../../../data/")
        #parser.add_argument("--raw_audio_base_path", type=str, default="../../../../data/coco_pyp/SpokenCOCO")
        parser.add_argument("--afiles", type=str, default="COCO")
        parser.add_argument("--vfiles", type=str, default="masked")
        parser.add_argument("--semtest_root", type=str, default="../../semtest/")
        parser.add_argument("--image_type", type=str, default="normal")
        parser.add_argument("--subset", type=str, default="all")
        parser.add_argument("--img_feat_len", type=int, help="num of img feats we will use", choices=list(range(1,37)), default=36)
        parser.add_argument("--audio_feat_len", type=float, help="maximal audio length", default=8.)
        parser.add_argument("--val_audio_feat_len", type=float, help="maximal audio length", default=10.)
        parser.add_argument("--coco_label_root", type=str, default="/data1/scratch/exp_pyp/MixedModal/hubert/coco")
        parser.add_argument("--normalize", action="store_true", default=False, help="whether or not normalize raw input, both w2v2 and hubert base doesn't normalize the input, but in exps in two papers, we normalized it, hopefully this doesn't make a difference")

    def __init__(self, args, split = "train"):
        self.args = args
        self.split = split
        self.audio_feat_len = args.audio_feat_len if "train" in split else args.val_audio_feat_len
        
        if split == "train":
            if args.subset == "all":
                # for original data
                print ('############# here is training on whole COCO data ###############')
                audio_dataset_json_file = os.path.join(args.data_root, "coco_pyp/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy.json")
            else:
                # for subsets
                print ('############# here is training on the ' + args.subset + ' data ###############')
                audio_dataset_json_file = '../../../../datavf/coco_pyp/subsets/SpokenCOCO_train_' + args.subset + '.json'
        elif split == "val" or split == "dev":
            if self.args.test:
                audio_dataset_json_file = os.path.join(self.args.semtest_root , 'data.json')
                # json file for semtest data
            else:
                audio_dataset_json_file = os.path.join(args.data_root, "coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json")
        
        
        self.audio_base_path = os.path.join(args.data_root, "coco_pyp/SpokenCOCO") 
        
        if args.image_type == "normal":
            # for otiginal images
            print ('############# here is training on normal images ###############')
            self.image_base_path = os.path.join(args.data_root, "coco_pyp/MSCOCO")
        elif args.image_type == "masked":
            print ('############# here is training on masked images ###############')
            if split == "train":
                self.image_base_path = os.path.join('../../../../datavf/', "coco_pyp/MSCOCO/masked" , args.subset)
            elif split == "val" or split == "dev":
                self.image_base_path = os.path.join(args.data_root, "coco_pyp/MSCOCO")
        elif args.image_type == "blurred":
            print ('############# here is training on blurred images ###############')
            if split == "train":
                self.image_base_path = os.path.join('../../../../datavf/', "coco_pyp/MSCOCO/blured" , args.subset)
            elif split == "val" or split == "dev":
                self.image_base_path = os.path.join(args.data_root, "coco_pyp/MSCOCO")
        
        # for masked and blured images:
        
        
        if "train" not in split:
            self.image_transform = transforms.Compose(
                [transforms.Resize(256, interpolation=Image.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.image_transform = transforms.Compose(
                [transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


        with open(audio_dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
            
        self.data = data_json['data']
        
        ##############################################
        if self.args.test:
            self.audio_base_path = os.path.join(self.args.semtest_root, self.args.afiles )
            self.image_base_path = os.path.join(self.args.semtest_root, self.args.vfiles )
        ##############################################


    def _LoadAudio(self, path):
        x, sr = sf.read(path, dtype = 'float32')
        assert sr == 16000
        length_orig = len(x)
        if length_orig > 16000 * self.audio_feat_len:
            audio_length = int(16000 * self.audio_feat_len)
            x = x[:audio_length]
            x_norm = (x - np.mean(x)) / np.std(x)
            x = torch.FloatTensor(x_norm) 
        else:
            audio_length = length_orig
            new_x = torch.zeros(int(16000 * self.audio_feat_len))
            x_norm = (x - np.mean(x)) / np.std(x)
            new_x[:audio_length] = torch.FloatTensor(x_norm) 
            x = new_x
        return x, audio_length

    def _LoadImage(self, impath):
        img = Image.open(impath).convert('RGB')
        img = self.image_transform(img)
        return img
    
    def __getitem__(self, index):
        datum = self.data[index]
        img_id = datum['image'].split("/")[-1].split(".")[0]
        #img_id = 'COCO_val2014_000000325114'
        wavpath = os.path.join(self.audio_base_path, datum['caption']['wav'])
        audio, nframes = self._LoadAudio(wavpath)  
        imgpath = os.path.join(self.image_base_path, datum['image'])
        img = self._LoadImage(imgpath)
        return img, audio, nframes, img_id, datum['caption']['wav']

    def __len__(self):
        return len(self.data)

    def collate(self, batch):
        #print('#################################')
        #print(batch)
        vals = list(zip(*batch))
        #print(vals)

        collated = {}
        # collated['visual_feats'] = torch.stack(vals[0])
        # collated['boxes'] = torch.stack(vals[1])
        
        collated['images'] = torch.stack(vals[0])
        collated['audio'] = torch.nn.utils.rnn.pad_sequence(vals[1], batch_first=True)
        collated['audio_length'] = torch.LongTensor(vals[2])
        collated['img_id'] = np.array(vals[3])
        collated['fn'] = vals[4]
        collated['audio_attention_mask'] = torch.arange(len(collated['audio'][0])).unsqueeze(0) >= collated['audio_length'].unsqueeze(1) 
        # kh mask = torch.arange(8).unsqueeze(0) >= torch.LongTensor(5).unsqueeze(1) 
        # print(mask) : above test creats a random boolean tensor 5*8 
        # print this to undestand what is happening here

        return collated

