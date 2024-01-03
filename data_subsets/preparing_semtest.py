import numpy
import cv2
import os
import json 
#from matplotlib import pyplot as plt
from utilsMSCOCO import read_data_from_path, get_all_cats, get_all_image_ids, change_labels
import copy

###############################################################################
                ############# masking images #############
###############################################################################
saveDir = "../../../semtest/images"
dataDir='../../data/coco_pyp/MSCOCO'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco , cats, cat_ids = read_data_from_path (dataDir, dataType)
cats_id_to_name, cats_id_to_supername, cats_name_to_id = get_all_cats (coco)

img_ids = get_all_image_ids (coco)

img_id_to_filenames = {}
img_filenames_to_id = {}
img_filenames_to_all = {}
for img_id in img_ids:
    img = coco.loadImgs(img_id)[0]
    fn = img['file_name']
    iD = img['id']
    img_id_to_filenames[iD] = fn
    img_filenames_to_id [fn] = iD
    img_filenames_to_all [fn] = img

all_labels, cats_id_to_short_name = change_labels (cats_id_to_name)   
#%%



def read_images_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data_json = json.load(json_file) 
    data = data_json['data']
    images = []
    for d in data:
        im = d['image']      
        images.append(im)
    return images
#%%
path_dict_obj_to_word = "../../datavf/coco_pyp/dict_words_selected.json"
with open(path_dict_obj_to_word, 'r', encoding='utf-8') as json_file:
    dict_obj_to_word = json.load(json_file)

dict_word_to_obj = {}
for key, value in dict_obj_to_word.items():
    for i in range (len(value)):
        dict_word_to_obj[value[i]] = key
    
#%%

path_test_json = "../../data/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json"
path_train_json_res = "../../datavf/coco_pyp/subsets/SpokenCOCO_train_SSL.json"

images_test = set (read_images_json(path_test_json)) #5 K

images_res = set (read_images_json(path_train_json_res)) #118 K

#%% finding image splits of wavs_res
images_res_train2014 = [] # 35 K
images_res_val2014 = [] # 82 K

for img_file in images_res:
    if 'train' in img_file:          
        images_res_train2014.append(img_file)
    elif 'val' in img_file:        
            images_res_val2014.append(img_file)


#%%
#subset test (5 K images)

# earlier used but the data is not enough

# residual subset (all train - subsets1,2,3,4)

images_test = images_res_val2014 # 35K

#%%

dict_images_test = {}

for im_path in images_test:
    
    if im_path not in dict_images_test:
        
        im_name = im_path.split('/')[1]
        img = img_filenames_to_all [im_name] 

        image_id = img ['id']
        h = img ['height']
        w = img ['width']
        name = img ['file_name']
        imPath = os.path.join(dataDir, dataType, name )
        image = cv2.imread(imPath)
        
        annId_img = coco.getAnnIds( imgIds=image_id, iscrowd=False) 
        anns_image = coco.loadAnns(annId_img)
        
        dict_images_test[im_path] = {}
        dict_images_test[im_path]['image_name'] = im_name
        dict_images_test[im_path]['image_id'] = image_id
        dict_images_test[im_path]['h'] = h
        dict_images_test[im_path]['w'] = w
        dict_images_test[im_path]['full_path'] = imPath
        dict_images_test[im_path]['annId_img'] = annId_img
        dict_images_test[im_path]['anns_image'] = anns_image
        cat_ids = []
        for item in anns_image:
            cat_ids.append(item['category_id'])
            
        dict_images_test[im_path]['cat_ids'] = cat_ids  

#%%
# For each category id, get images
dict_id_to_image = {}

for key, value in dict_images_test.items():
    cat_ids = list(set(value['cat_ids']))
    print(cat_ids)
    for objID in cat_ids:
        if objID not in dict_id_to_image:
            dict_id_to_image[objID] = []
            dict_id_to_image[objID].append(key)
        else:
            dict_id_to_image[objID].append(key)

#%%
# step 1: selecting 100 candidates from each category
# step 2: detecting obj mask and res mask and sorting 1000 candidates based on the least overlap
# step 3: measuring the average area and setting a minumum of 0.1*average area
# step 4: selecting the first 20 candidates that pass the condition of minimum area based on iterative search (1, 0.5, 0.1)

#%% 
# measure overlap of objects

dict_obj_sorted_image = {}
dict_areas = {}
for key, value in dict_id_to_image.items():
    objID = key
    dict_obj_sorted_image[objID] = {}
    temp_image_names = []
    temp_overlaps = []
    temp_obj_area_ratio = []
    for image_candidate in value [0:1000]: #selecting first 1000 candidates for each category since this is unnecessarily huge for some categories
        
        info = dict_images_test [image_candidate]
        # index_objID = [i for i in range(len(info['cat_ids'])) if info['cat_ids'][i] == objID ]
        # index_rest = [i for i in range(len(info['cat_ids'])) if info['cat_ids'][i] != objID ]
        h = info['h']
        w = info['w']
        area = h*w
        anns_image = info['anns_image']
        mask_obj = numpy.zeros([h,w])
        mask_rest = numpy.zeros([h,w])
        for item in anns_image : # constructing true mask by ading all mask items
            mask_temp = coco.annToMask(item)
            if item['category_id'] == objID:
                mask_obj = mask_obj + mask_temp
            else:
                mask_rest = mask_rest + mask_temp
        # converting the mask to 0 and 1 binary values        
        mask_binary_obj = 1 * (mask_obj > 0 )
        mask_binary_rest = 1 * (mask_rest > 0 )
        
        area_object = sum(sum(mask_binary_obj))
        obj_area_ratio = round(area_object/area,4)
        # we add both binary masks,
        # the number of elements bigger than 1 will show the number of overlapping pixels
        mask_all = mask_binary_obj + mask_binary_rest
        mask_overlap = 1 * (mask_all > 1 )      
        overlap = sum(sum(mask_overlap))
        #print(overlap)
        #if obj_area_ratio >= 0.05:
        temp_image_names.append(image_candidate)
        temp_overlaps.append(overlap)
        temp_obj_area_ratio.append(obj_area_ratio)
    # sorting overlaps and save the image list based on sorted values    
    sorted_args = numpy.argsort(temp_overlaps) 
    temp_image_names_sorted = [temp_image_names[i] for i in sorted_args]
    temp_obj_area_ratio_sorted = [temp_obj_area_ratio[i] for i in sorted_args]
    dict_obj_sorted_image[objID]['areas'] = temp_obj_area_ratio_sorted
    dict_obj_sorted_image[objID]['images'] = temp_image_names_sorted
    dict_areas[objID] = numpy.mean(temp_obj_area_ratio_sorted)
   
#%%

# record which iObject category does not have 20 candidates
# ball, baseball, glove, dryer (37, 39, 40, 80, 89)
dict_poor_candidates = {}
for key, dict_images in dict_obj_sorted_image.items():
    objID = key
    images_list = dict_images ['images']
    if len(images_list) < 20:
        dict_poor_candidates [objID] = len(images_list)
        
#%% next, read images from "dic_obj_sorted_image"   and for each key (objID) select the first 20 candidates
# and save the masked version of those candidates where the mask is only applied on the ID objects
dict_saved_images = {}     
for key, dict_images in dict_obj_sorted_image.items():
    objID = key
    images_list = dict_images ['images']
    obj_area_list = dict_images ['areas']
    average_area = dict_areas[objID]
    
    # select as big objects as possible based on object characteristics
    args_area = [ind for ind in range(len(obj_area_list)) if obj_area_list[ind] > average_area * 1]
    if len(args_area) < 20:
        args_area = [ind for ind in range(len(obj_area_list)) if obj_area_list[ind] > average_area * 0.5]
    if len(args_area) < 20:
        args_area = [ind for ind in range(len(obj_area_list)) if obj_area_list[ind] > average_area * 0.1]    
    
    images_list = [images_list [i] for i in args_area]
    
    sem_testset = images_list [0:20]
    dict_saved_images [objID] = {}
    dict_saved_images [objID]['image_list'] = sem_testset
    
    saved_names = []
    area_ratios = []
    categories = []
    for counter, image_candidate in enumerate(sem_testset):
        
        info = dict_images_test [image_candidate]
        h = info['h']
        w = info['w']
        anns_image = info['anns_image']
        mask_obj = numpy.zeros([h,w])
        
        name = info ['image_name']
        imPath = os.path.join(dataDir, dataType,name )
        image = cv2.imread(imPath)
        
        for item in anns_image : # constructing true mask by ading all mask items
            mask_temp = coco.annToMask(item)
            if item['category_id'] == objID:
                mask_obj = mask_obj + mask_temp
            # else:
            #     mask_rest = mask_rest + mask_temp
        # converting the mask to 0 and 1 binary values        
        mask_binary_obj = 1 * (mask_obj > 0 )
        # mask_binary_rest = 1 * (mask_rest > 0 )
      
        mask_binary_uint = numpy.array(mask_binary_obj, dtype=numpy.uint8)
        mask = 1 * mask_binary_uint
        ################################## masking the image
        masked_image = copy.deepcopy (image)
        masked_image[:,:,0] = image[:,:,0] * mask
        masked_image[:,:,1] = image[:,:,1] * mask
        masked_image[:,:,2] = image[:,:,2] * mask
        ################################## masking the blurred image
        blured_image = cv2.blur(image,(30,30),0)
        masked_image_blur = copy.deepcopy (blured_image)
        masked_image_blur[:,:,0] = blured_image[:,:,0] * (1- mask)
        masked_image_blur[:,:,1] = blured_image[:,:,1] * (1- mask)
        masked_image_blur[:,:,2] = blured_image[:,:,2] * (1- mask)
        blurmasked_image = masked_image_blur + masked_image 
        ##################################
        
        cat_obj = cats[objID]['name']
        #saving using the short name
        label_obj = cats_id_to_short_name[objID]
        area_obj = obj_area_list [counter]
        save_name = label_obj + '_' + str(counter + 1) + '.jpg'
        
        save_path_masked = os.path.join(saveDir, 'masked', save_name )
        save_path_blurred = os.path.join(saveDir, 'blurred', save_name )
        save_path_original = os.path.join(saveDir, 'original', save_name )
        
        # cv2.imwrite(save_path_masked , masked_image)  
        # cv2.imwrite(save_path_blurred , blurmasked_image)
        # cv2.imwrite(save_path_original , image)
        
        saved_names.append(save_name)
        area_ratios.append(area_obj)
        
    dict_saved_images [objID]['saved_names'] = saved_names    
    dict_saved_images [objID]['area_ratios'] = area_ratios 
    dict_saved_images [objID]['categories'] = cat_obj
    dict_saved_images [objID]['label'] = label_obj
    
# file_json = "/worktmp2/hxkhkh/current/semtest/images/" + "semtest_images.json"
# with open(file_json, "w") as fp:
#     json.dump(dict_saved_images,fp) 
#%% testing json file
import json
file_json = "/worktmp2/hxkhkh/current/semtest/images/" + "semtest_images.json"
with open(file_json, 'r', encoding='utf-8') as json_file:
    data_json_saved = json.load(json_file) 

#%%
dict_snames_to_imnames = {}
for key, value in data_json_saved.items():
    imlist = value['image_list']
    snames = value['saved_names']
    for counter, imn in enumerate(imlist):
        ims = snames[counter]
        dict_snames_to_imnames [ims] = imn
#%%   
import json   
file_json = '/worktmp2/hxkhkh/current/semtest/data.json'
with open(file_json, 'r', encoding='utf-8') as json_file:
    data_DINO = json.load(json_file) ['data']

list_RCNN = []
for item in data_DINO:
    i = item['image']
    c = item['caption']
    item_rcnn = {}
    item_rcnn['image'] = dict_snames_to_imnames [i]
    item_rcnn['caption'] = c
    
    list_RCNN.append(item_rcnn)

data_RCNN = {}
data_RCNN['data'] = list_RCNN
#%%
# file_json = '/worktmp2/hxkhkh/current/semtest/dataRCNN.json'
# with open(file_json, "w") as fp:
#     json.dump(data_RCNN,fp) 