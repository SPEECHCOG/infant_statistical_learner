
import json
import os
import shutil
#%% below I am testing original utterances
# 1 we need data on image names of semtest
# 2 we need to extract audio file names given each image names from source data (res SSL) 
# 3 we select 1 utterance for each test image
# 4 we save the new extracted data in info_semtest to know what samples we use for semtest from original COCO
# 5 we save the utterances in semtest/utterances folder using lexical namings

#%%
# 1
file_original_images= '../../../semtest/images/semtest_images.json'
with open(file_original_images, 'r', encoding='utf-8') as json_file:
    images_semtest = json.load(json_file) 

# 2
path_train_json_res = "../../datavf/coco_pyp/subsets/SpokenCOCO_train_SSL.json"
with open(path_train_json_res, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file) 
data_res = data['data']

# 3
dict_image_to_wav_res = {}
for item in data_res:
    image = item['image']
    wav_image = item ['caption']['wav']
    if image not in dict_image_to_wav_res:
        dict_image_to_wav_res [image] = wav_image
# 4
info_semtest = {}
for cat, info_cat in images_semtest.items():
    info_semtest [cat] = {}
    
    image_list = info_cat['image_list']
    names_list = info_cat['saved_names']
    
    wavs_list = []
    for img in image_list:
        wav_im = dict_image_to_wav_res[img]
        wavs_list.append(wav_im)
        
    info_semtest[cat]['images'] = image_list
    info_semtest[cat]['names'] = names_list
    info_semtest[cat]['utterances'] = wavs_list
#%%
# 5

file_json_pairings =  "../../../semtest/semtest_files_pairings.json"  
with open(file_json_pairings, 'r', encoding='utf-8') as json_file:
    data_pairings_word_to_object = json.load(json_file)

data_pairings_object_to_word = {}
for key, value in data_pairings_word_to_object.items():
    data_pairings_object_to_word[value] = key
    
#%%

dataDir_img = '../../data/coco_pyp/MSCOCO/'
saveDir_img = '/worktmp2/hxkhkh/current/semtest/images/original/'

dataDir_wav = '../../data/coco_pyp/SpokenCOCO/'
saveDir_wav = '/worktmp2/hxkhkh/current/semtest/utterances/'


dict_utterances = {}
for cat, info in info_semtest.items():
    images = info['images']
    utterances = info['utterances']
    names_imgs = info['names']
    for counter, uttr in enumerate(utterances):
        
        img = images[counter]
        name_img = names_imgs [counter]
        file_origin_img = os.path.join(dataDir_img, img)
        file_target_img = os.path.join(saveDir_img, name_img)
        shutil.copy(file_origin_img, file_target_img)
        
        file_origin_wav = os.path.join(dataDir_wav, uttr)
        name_wav =  data_pairings_object_to_word[name_img]
        file_target_wav = os.path.join(saveDir_wav, name_wav)
        shutil.copy(file_origin_wav, file_target_wav)
        
 