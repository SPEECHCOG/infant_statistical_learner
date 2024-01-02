#%%
import json

# getting SpokenCOCO-SSL subset by removing the largest audiovisual subset from data (sub 3 + uniform 0A)
# note that this discards all audiovisual training data since sub3 (6 months) covers other smallers subsets.

# reading the whole data

audio_dataset_json_file = '/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy.json'
with open(audio_dataset_json_file, 'r') as fp:
    data_json = json.load(fp)
data = data_json['data']


# reading audiovisual subsets 

subset_name = 'subset3'
file_json = "/worktmp2/hxkhkh/current/FaST/datavf/coco/subsets/SpokenCOCO_train_" + subset_name +  ".json"
with open(file_json, 'r') as fp:
    data_json_vgs = json.load(fp)
    
data_subset3_vgs = data_json_vgs['data']


subset_name = 'subset0A'
file_json = "/worktmp2/hxkhkh/current/FaST/datavf/coco/subsets/SpokenCOCO_train_" + subset_name +  ".json"
with open(file_json, 'r') as fp:
    data_json_vgs = json.load(fp)
    
data_subset0A_vgs = data_json_vgs['data']

#%%


# saving data SSL json file

data_subset_SSL = []    
for d in data:
    if d not in data_subset3_vgs and d not in data_subset0A_vgs:
        data_subset_SSL.append(d)
        
data_json_SSL = {}
data_json_SSL ['data'] = data_subset_SSL
file_json = "/worktmp2/hxkhkh/current/FaST/datavf/coco/subsets/SpokenCOCO_train_SSL.json"
with open(file_json, "w") as fp:
    json.dump(data_json_SSL,fp) 


#%%

# checking the saved file
import json
    
subset_name = 'SSL'
file_json = "/worktmp2/hxkhkh/current/FaST/datavf/coco/subsets/SpokenCOCO_train_" + subset_name +  ".json"
with open(file_json, 'r') as fp:
    data_json_test = json.load(fp)
    
data_subset_test = data_json_test['data']
    
print(len(data_subset_test))
print(len(data_subset_test)/64)


######### to measure the speech time
import soundfile as sf
import os 
path_wav = '/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO/'
seconds_orig = []
seconds_applied = []
for d in data_subset_test:
    audiofile = d['caption']['wav']
    path = os.path.join(path_wav,audiofile)
    x, sr = sf.read(path, dtype = 'float32')
    length_orig = len(x)
    time_orig = length_orig /sr
    seconds_orig.append(time_orig)
    
    if length_orig > sr * 8:
        seconds_applied.append(8)
    else:
        seconds_applied.append(time_orig)
    
hours = sum(seconds_orig)/3600
print(' ..... total time is ....' + str(hours))

hours_applied = sum(seconds_applied)/3600
print(' ..... total time is ....' + str(hours_applied))


#%%
######### to get statistics of COCO

audio_dataset_json_file = '/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy.json'
with open(audio_dataset_json_file, 'r') as fp:
    data_json = json.load(fp)
data = data_json['data']

train_size = len (data)


audio_dataset_json_file = '/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json'
with open(audio_dataset_json_file, 'r') as fp:
    data_json = json.load(fp)
data = data_json['data']

val_size = len (data)

total_size = train_size + val_size
total_time = 742 # hours
size_per_hour = round(total_size / total_time )
seconds_per_utt = round ((total_time/total_size) * 3600 , 2)

#%%
########## to copy images and speech of subsets 

import json
import os
import shutil    

path_images = '/worktmp2/hxkhkh/current/FaST/data/coco_pyp/MSCOCO/'
path_wav = '/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO/'

dest_images = '/worktmp2/hxkhkh/current/FaST/data/coco_example/subset1/images/'
dest_wavs = '/worktmp2/hxkhkh/current/FaST/data/coco_example/subset1/wavs/'

subset_name = 'subset1'
file_json = "/worktmp2/hxkhkh/current/FaST/data/coco/subsets/SpokenCOCO_train_" + subset_name +  ".json"
with open(file_json, 'r') as fp:
    data_json_test = json.load(fp)
    
data_subset_test = data_json_test['data']
for counter, item in enumerate(data_subset_test):
    image = item['image']
    wav = item['caption']['wav']
    image_file = os.path.join(path_images, image)
    wav_file = os.path.join(path_wav, wav)
    
    # use names or replace with index
    # im = (image.split('/'))[-1]
    # w = (wav.split('/'))[-1]
    # print(im)
    # print(w)
    im = str(counter) + '.jpg'
    w = str(counter) + '.wav'
    image_dest_file = os.path.join(dest_images, im)
    wav_dest_file = os.path.join(dest_wavs, w)
    shutil.copy(image_file, image_dest_file)
    shutil.copy(wav_file, wav_dest_file)
    