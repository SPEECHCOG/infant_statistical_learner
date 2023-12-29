import numpy as np
import cv2
import os
import json 
from utilsMSCOCO import read_data_from_path, get_all_cats, get_all_image_ids, change_labels
root = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/subsets/"
jsub0 = os.path.join(root, "SpokenCOCO_train_subset0A.json")
jsub1 = os.path.join(root, "SpokenCOCO_train_subset1.json")
jsub2 = os.path.join(root, "SpokenCOCO_train_subset2.json")
jsub3 = os.path.join(root, "SpokenCOCO_train_subset3.json")
#%%

def read_data_subsets(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data_json = json.load(json_file) 
    data = data_json['data']
    return data

# stack all data
data0 = read_data_subsets(jsub0)
data1 = read_data_subsets(jsub1)
data2 = read_data_subsets(jsub2)
data3 = read_data_subsets(jsub3)

#%%
def find_image_unique_labels (imID, coco):
    annId_img = coco.getAnnIds( imgIds=imID, iscrowd=False) 
    anns_image = coco.loadAnns(annId_img)
    unique_objects_ids = []
    for item in anns_image:
        k = item['category_id']
        #m = cats_id_to_name[k]
        if k not in unique_objects_ids:
            unique_objects_ids.append(k)
            
    return unique_objects_ids       

def find_image_path (imID, coco, dataType):
    img = coco.loadImgs(imID)[0]      
    name = img ['file_name']
    imPath = os.path.join(dataType,name )
    # imFullPath = os.path.join(dataDir, dataType_train,name )
    # image = cv2.imread(imFullPath)        
    # plt.imshow(image)
    return imPath

def find_dict_image_to_label (coco, dataType, img_ids ):
    dict_img_id_to_path = {}
    dict_img_path_to_id = {}
    dict_image_to_label = {}
    
    for ind in range(len(img_ids)):
        imID = img_ids[ind]
        # print (ind)
        # print(img_ids)
        unique_objects_ids = find_image_unique_labels (imID, coco)  
        dict_image_to_label [imID] = unique_objects_ids
        
        imPath = find_image_path (imID, coco, dataType)
        dict_img_id_to_path [imID] = imPath
        dict_img_path_to_id [imPath] = imID
    return dict_image_to_label, dict_img_id_to_path, dict_img_path_to_id

saveDir = "../../../semtest/images"
dataDir='../../data/coco_pyp/MSCOCO'
dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco , cats, cat_ids = read_data_from_path (dataDir, dataType)
cats_id_to_name, cats_id_to_supername, cats_name_to_id = get_all_cats (coco)

img_ids = get_all_image_ids (coco)

img_id_to_img_filenames = {}
img_filenames_to_img_id = {}
img_filenames_to_all = {}
for img_id in img_ids:
    img = coco.loadImgs(img_id)[0]
    fn = img['file_name']
    imID = img['id']
    img_id_to_img_filenames[imID] = fn
    img_filenames_to_img_id [fn] = imID
    img_filenames_to_all [fn] = img
    

all_labels, cats_id_to_short_name = change_labels (cats_id_to_name)  
dict_image_to_label_train, dict_img_id_to_path_train, dict_img_path_to_id_train = find_dict_image_to_label (coco, dataType,img_ids )
dict_image_to_label_all = dict_image_to_label_train
#%% copied this for obtaining "label_word_sorted" but it can be accesses also through dict_words_selected

caption_json = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/dict_words_selected.json"
with open(caption_json, 'r') as fp:
    dict_words_selected = json.load(fp)

caption_json = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/dict_words_selected_counts.json"
with open(caption_json, 'r') as fp:
    dict_words_selected_counts = json.load(fp)
   
import copy
# manually filtering words
dict_words_selected_filtered = copy.deepcopy(dict_words_selected)

dict_words_selected_filtered ['baseball'] = ['baseball bat']
dict_words_selected_filtered ['parking'] = ['parking meter']
dict_words_selected_filtered ['wineglass'] = ['wine glass']
dict_words_selected_filtered ['traffic'] = ['traffic light']
dict_words_selected_filtered ['stop'] =  ['stop sign']
dict_words_selected_filtered ['glove'] = ['baseball glove', 'baseball gloves']
dict_words_selected_filtered ['handbag'] = ['purse'] # this should be changed to handbag later
dict_words_selected_filtered ['dryer'] = ['hair dryer']


frequent_counts_sorted = sorted (dict_words_selected_counts.items(), key=lambda x:x[1], reverse=True)

words_sorted = []
labels_sorted = []
for tuple_word_count in frequent_counts_sorted:   
    labels_sorted.append(tuple_word_count[0])
    words_sorted.append(dict_words_selected_filtered[tuple_word_count[0]])
    
label_word_sorted = []
for counter, l in enumerate(labels_sorted):
    lw_pair = (l, words_sorted[counter])
    label_word_sorted.append(lw_pair)
caption_json = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/dict_sorted_objects.json"
with open(caption_json, 'r') as fp:
    labels_sorted = json.load(fp) ['sorted_object_labels']
    
label_word_sorted = []
for counter, l in enumerate(labels_sorted):
    lw_pair = (l, words_sorted[counter])
    label_word_sorted.append(lw_pair)

dict_label_to_word_final_list = {}
for item in label_word_sorted:
    dict_label_to_word_final_list[item[0]] = item[1]          
#%% select a subset and run the code for it

data_sub = data0
name_meta = 'nsub0'
dict_counts_sub = {}
dict_areas_sub = {}
dict_captions_sub = {}
dict_images_sub = {}
# apple_anns_image = []
# apple_areas = []
for d in data_sub:
    #print('...........................................................')
    im_path = d['image']
    im_name = im_path.split('/')[1]
    imID = img_filenames_to_img_id [im_name] 
    h = img_filenames_to_all[im_name]['height']
    w = img_filenames_to_all[im_name]['width']
    areaI = h*w
    annId_img = coco.getAnnIds( imgIds=imID, iscrowd=False) 
    anns_image = coco.loadAnns(annId_img)
    objs = {}
    for annitem in anns_image:
        l = cats_id_to_name [annitem ['category_id'] ]
        # ratio to all image
        a = annitem ['area'] / areaI
        if l not in objs:
            objs[l] = [a]
        else:
            objs[l].append(a)
   
    cap = d['caption']['text']
    cap = cap.lower()
    #print(cap)
    for obj,list_areas  in objs.items():
        names_obj = dict_label_to_word_final_list[obj]
        for n in names_obj:
            w = ' ' + n + ' '
            if w in cap:
                
                # a hit is detected
                mean_area = np.mean(list_areas)
                
                if obj not in dict_areas_sub:
                    dict_areas_sub[obj] = [mean_area]
                else:
                    dict_areas_sub[obj].append(mean_area)
                
                pair = (obj, n)
                # if obj == "apple":
                #     print(pair)
                    
                #     imPath = os.path.join(dataDir, dataType, im_name )
                #     image = cv2.imread(imPath)
                #     apple_anns_image.append(anns_image)
                #     apple_areas.append(mean_area)
                if obj in dict_counts_sub:
                    dict_counts_sub[ obj] += 1
                    dict_captions_sub [obj].append(cap)
                    dict_images_sub [obj].append(im_path)
                    #print(pair)
                    
                else:
                    dict_counts_sub[ obj] = 1
                    dict_captions_sub [obj] = []
                    dict_captions_sub [obj].append(cap)
                    dict_images_sub [obj] = []
                    dict_images_sub [obj].append(im_path)
                    #print(pair)
                    
dict_mean_areas_sub = {}                      
for key, value in dict_areas_sub.items():
    dict_mean_areas_sub[key] = np.mean(value)

dict_mean_areas_sub_sorted = sorted (dict_mean_areas_sub.items(), key=lambda x:x[1], reverse=True)    

### saving meta data
data_save = {}
data_save['object_areas'] = dict_mean_areas_sub
data_save['object_freq'] = dict_counts_sub
data_save['object_cap'] = dict_captions_sub
file_json = os.path.join(root, name_meta + '_meta.json' )
with open(file_json, "w") as fp:
    json.dump(data_save,fp) 



#%% testing the saved files
f = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/subsets/sub3_meta.json"
with open(f, 'r') as fp:
    dict_test = json.load(fp)
freqs = sorted (dict_test['object_freq'].items(), key=lambda x:x[1], reverse=True)
# (min, max) frequencies
# 8 months (2,89)
# 10 months (4, 178)
# 10 months uniform (37,37)
# 12 months (6, 267)