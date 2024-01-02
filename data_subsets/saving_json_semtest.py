#%%
#this is the process of saving Semtest test data in json karapathy format
#%% below is related to reading test data
import json
import os
data_root = '/worktmp2/hxkhkh/current/FaST/data'
audio_dataset_json_file = os.path.join(data_root, "coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json")
with open(audio_dataset_json_file, 'r', encoding='utf-8') as json_file:
    data_json = json.load(json_file) 
data = data_json['data']
# image, caption (text, speaker, uttid, wav)
img_example = data[0]['image']
wav_example = data[0]['caption']['wav']
print(img_example)
print(wav_example)
#val2014/COCO_val2014_000000325114.jpg
#wavs/val/0/m071506418gb9vo0w5xq3-3LUY3GC63Z0R9PYEETJGN5HO4UEP7B_325114_629297.wav

datum = data[0]#[index]
#img_id = datum['image'].split("/")[-1].split(".")[0] ----> 'COCO_val2014_000000325114'

#wavpath = os.path.join(self.audio_base_path, datum['caption']['wav'])
#imgpath = os.path.join(self.image_base_path, datum['image'])

# audio, nframes = self._LoadAudio(wavpath)
# img = self._LoadImage(imgpath)

# self.audio_base_path = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO"
# self.image_base_path = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/MSCOCO"
# for semtest:
# audio_base_path = "/worktmp2/hxkhkh/current/semtest/utterances/"
# image_base_path = "/worktmp2/hxkhkh/current/semtest/images/original/"

#%%
save_path = '/worktmp2/hxkhkh/current/semtest/'
file_json_pairings =  "../../semtest/semtest_files_pairings.json"  
# reading test datafile names 
with open(file_json_pairings, 'r', encoding='utf-8') as json_file:
    data_pairings = json.load(json_file) 

wav_files = []
img_files = []
data_list = []
for w, i in data_pairings.items():
    wav_files.append(w)
    img_files.append(i)
    d = {}
    d['image'] = i
    d['caption'] = {} 
    d['caption']['wav'] = w
    data_list.append(d)

data_list[0]['caption']['wav']
data_dict = {}
data_dict['data'] = data_list 

#%%
file_json = "/worktmp2/hxkhkh/current/semtest/data.json"
with open(file_json, "w") as fp:
    json.dump(data_dict,fp)
#%% 
