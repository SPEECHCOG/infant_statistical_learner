
import os
import json
from utilsMSCOCO import *

###############################################################################
                ############# COCO aux functions #############
###############################################################################


    
    
def create_subset (objects_sorted, values_sorted, dataType, coco, k, n):
    subset = {}
    objects = objects_sorted[0:k]
    cats_id_to_name, cats_id_to_supername, cats_name_to_id = get_all_cats (coco)
    #values_1 = values_sorted[0:10]
    images = []
    image_files = []     
    for obj in objects:
        objID = cats_name_to_id[obj]
        img_ids_query = coco.getImgIds(catIds=[objID])[0:n]
        subset[obj] = {}
        subset[obj]['objID'] = objID
        subset[obj]['images'] = img_ids_query
        for item in img_ids_query [0:n]:
            if item not in images:
                images.append(item)
                img_info = coco.loadImgs([item])[0]
                image_files.append(os.path.join(dataType , img_info["file_name"]))
    return subset, objects , image_files      

def creating_final_subsets (dataDir, dataType , k, n):
    
    coco, cats, cat_ids = read_data_from_path (dataDir, dataType)
    all_counts_images, all_images_cats, all_images_supercats =  get_images_per_cats (cat_ids, coco) 
     
    objects = list(all_counts_images.keys())
    values = list (all_counts_images.values())
    sorted_ind, objects_sorted, values_sorted = sort_object (objects, values)    
    subset, objects , image_files = create_subset (objects_sorted, values_sorted, dataType, coco, k, n)
    return subset, objects , image_files

def save_json_subsets (data_root, image_files_train, jname_orig, jname_sub):

    audio_dataset_json_file = os.path.join(data_root, jname_orig)
    with open(audio_dataset_json_file, 'r') as fp:
        data_json = json.load(fp)
    data = data_json['data']
    
    data_sub = [] 
    for item in data:
        imfile = item['image']
        if imfile in image_files_train:
            data_sub.append(item)
            
    # writing json file        
    dictionary = {}
    dictionary ['data'] = data_sub
    json_object = json.dumps(dictionary, indent=4)
    json_file = os.path.join(data_root, jname_sub ) 
    # Writing to sample.json
    with open(json_file, "w") as outfile:
        outfile.write(json_object)
    
    return data_sub

#############################################
# plotting the distribution of the objects
#############################################

######################### reading train and val data

# dataDir='../data/coco_pyp/MSCOCO'
# dataType='train2014'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# coco_train, cats, cat_ids = read_data_from_path (dataDir, dataType)
# all_counts_images_train, all_images_cats_train, all_images_supercats_train =  get_images_per_cats (cat_ids, coco_train) 

# dataDir='../data/coco_pyp/MSCOCO'
# dataType='val2014'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# coco_val, cats, cat_ids = read_data_from_path (dataDir, dataType)
# all_counts_images_val, all_images_cats_val, all_images_supercats_val =  get_images_per_cats (cat_ids, coco_val) 

# ######################## unifying train and val sets

# all_counts_images = {}
# all_images_cats = {}
# all_images_supercats = {}
# for key in all_counts_images_train:
#     all_counts_images [key] = all_counts_images_train [key] + all_counts_images_val [key]
#     all_images_cats [key] = all_images_cats_train [key] + all_images_cats_val [key]
# for key in all_images_supercats_train:
#     all_images_supercats [key] = all_images_supercats_train [key] + all_images_supercats_val [key]

# ######################## plotting 

# save_path = '/worktmp2/hxkhkh/current/FaST/experiments/plots/vf/distributions/'
# save_plot (save_path + 'all', all_counts_images, all_images_supercats )
# save_plot (save_path + 'train', all_counts_images_train, all_images_supercats_train )
# save_plot (save_path + 'val', all_counts_images_val, all_images_supercats_val )

###############################################################################
                ############# creating subsets #############
###############################################################################

######################## checking if in Karpathy train split, there is any val file

# data_root = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp"
# audio_dataset_json_file = os.path.join(data_root, "SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json")

# with open(audio_dataset_json_file, 'r') as fp:
#     data_json = json.load(fp)
    
# data = data_json['data']
# count_train = 0
# for item in data:
#     namef = item['image'].split('/')[0]
#     if 'train' in namef:
#         count_train += 1
        
# Results is zero; there is no val file in Karpathy train split
# So, we select out subset only from COCO train set, sinc the size is enough

######################### creating train subsets #############################

######################### reading train data

dataDir='../data/coco_pyp/MSCOCO'
dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    
# subset 1: small subset (10 most frequent objects)
##################################################
k1 = 10 
n1 = 150 # how many images for each object
subset_train_1, objects_train_1 , image_files_train_1 = creating_final_subsets (dataDir, dataType , k1, n1)
# subset 2: medium subset (40 most frequent objects)
###################################################
k2 = 40 
n2 = 150 # how many images for each object
subset_train_2, objects_train_2 , image_files_train_2 = creating_final_subsets (dataDir, dataType , k2, n2)


######################### creating val subsets #############################

######################### reading val data

dataDir='../data/coco_pyp/MSCOCO'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    
# subset 1: small subset (10 most frequent objects)
##################################################
k1 = 10 
n1 = -1 # how many images for each object
subset_val_1, objects_val_1 , image_files_val_1 = creating_final_subsets (dataDir, dataType , k1, n1)
# subset 2: medium subset (40 most frequent objects)
###################################################
k2 = 40 
n2 = -1 # how many images for each object
subset_val_2, objects_val_2 , image_files_val_2 = creating_final_subsets (dataDir, dataType , k2, n2)

###############################################################################
                ############# saving subsets #############
###############################################################################
 
data_root = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp"

# for train split

jname_orig = "SpokenCOCO/SpokenCOCO_train_unrolled_karpathy.json"

jname_sub = "SpokenCOCO/SpokenCOCO_train_sub1_unrolled_karpathy.json"
data_sub_train_1 = save_json_subsets (data_root, image_files_train_1, jname_orig, jname_sub)

jname_sub = "SpokenCOCO/SpokenCOCO_train_sub2_unrolled_karpathy.json"
data_sub_train_2 = save_json_subsets (data_root, image_files_train_2, jname_orig, jname_sub)

# for val split 

jname_orig = "SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json"

jname_sub = "SpokenCOCO/SpokenCOCO_val_sub1_unrolled_karpathy.json"
data_sub_val_1 = save_json_subsets (data_root, image_files_val_1, jname_orig, jname_sub)

jname_sub = "SpokenCOCO/SpokenCOCO_val_sub2_unrolled_karpathy.json"
data_sub_val_2 = save_json_subsets (data_root, image_files_val_2, jname_orig, jname_sub)
