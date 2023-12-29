import os
import json
#%%
# step 1. readng all files from semtest audio and image paths
# step 2. exctracting the "pure" name used for objects and words
# step 4. reading info from json file on paired object names and spoken words
#%% step 1
audio_path = "../../../semtest/COCO/"
visual_path = "../../../semtest/images/masked/"


wav_files_mixed = os.listdir(audio_path)
img_files_mixed = os.listdir(visual_path)

#%% step 2
wav_names_mixed = []
img_names_mixed = []

for wav in wav_files_mixed:
    wav_names_mixed.append(wav.split('_')[0])
for img in img_files_mixed:
    img_names_mixed.append(img.split('_')[0])
#%% step 3

path_dict_obj_to_word = "../../datavf/coco_pyp/dict_words_selected.json"
with open(path_dict_obj_to_word, 'r', encoding='utf-8') as json_file:
    dict_obj_to_word = json.load(json_file)

dict_word_to_obj = {}
for key, value in dict_obj_to_word.items():
    for i in range (len(value)):
        dict_word_to_obj[value[i]] = key
        
#%% step 4
wav_files_sorted = []
img_files_sorted = []
dict_pairings = {}
for spoken_word in dict_word_to_obj:
    if spoken_word in wav_names_mixed:
        obj_name = dict_word_to_obj [spoken_word]
        
        for counter in range(1,21):
            sw = spoken_word + '_' + str(counter) + '.wav'
            obn = obj_name + '_' + str(counter) + '.jpg' 
            wav_files_sorted.append(sw) 
            img_files_sorted.append(obn)
            dict_pairings [sw] = obn
#%% step 5 save dictionary of file name pairings 
file_json_pairings = "../../../semtest/" + "semtest_files_pairings.json"
with open(file_json_pairings, "w") as fp:
    json.dump(dict_pairings,fp)      


#%% testing the json file

with open(file_json_pairings, 'r', encoding='utf-8') as json_file:
    data_pairings = json.load(json_file) 

#%%

