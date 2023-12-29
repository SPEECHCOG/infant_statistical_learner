
import json
import os

from utilsMSCOCO import read_data_from_path, get_all_image_ids#, get_all_cats

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
    img_id_to_path = {}
    img_path_to_id = {}
    dict_image_to_label = {}
    
    for ind in range(len(img_ids)):
        imID = img_ids[ind]
        # print (ind)
        # print(img_ids)
        unique_objects_ids = find_image_unique_labels (imID, coco)  
        dict_image_to_label [imID] = unique_objects_ids
        
        imPath = find_image_path (imID, coco, dataType)
        img_id_to_path [imID] = imPath
        img_path_to_id [imPath] = imID
    return dict_image_to_label, img_id_to_path, img_path_to_id

#%%


dataDir='../../data/coco_pyp/MSCOCO'
dataType_train='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType_train)
coco_train, cats, cat_ids = read_data_from_path (dataDir, dataType_train)

dataType_val='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType_val)
coco_val, cats, cat_ids = read_data_from_path (dataDir, dataType_val)

img_ids_train = get_all_image_ids (coco_train)
img_ids_val = get_all_image_ids (coco_val)


# For the train split   
dict_image_to_label_train, img_id_to_path_train, img_path_to_id_train = find_dict_image_to_label (coco_train, dataType_train ,img_ids_train ) 

# For the validation split
  
dict_image_to_label_val, img_id_to_path_val, img_path_to_id_val = find_dict_image_to_label (coco_val, dataType_val ,img_ids_val ) 

# merging train and val

dict_image_to_label_all = dict_image_to_label_train #{**dict_image_to_label_train, **dict_image_to_label_val}
  
img_id_to_path_all = {**img_id_to_path_train , **img_id_to_path_val}
img_path_to_id_all = {**img_path_to_id_train , **img_path_to_id_val}

#%%

caption_json = "/worktmp2/hxkhkh/current/FaST/data/coco_subsets/captions_train2014_subset1.json"
with open(caption_json, 'r') as fp:
    data_json_subset = json.load(fp)
data_subset =  data_json_subset ['annotations']


audio_dataset_json_file = '/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy.json'
with open(audio_dataset_json_file, 'r') as fp:
    data_json = json.load(fp)
data_train = data_json['data']

audio_dataset_json_file = '/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json'
with open(audio_dataset_json_file, 'r') as fp:
    data_json = json.load(fp)
data_val = data_json['data']

data = data_train + data_val

#%%    
def modify_cap (cap):
    cap_modified = cap
    if cap[0] == ' ' :
        cap_modified = cap[1:]
    if cap[-1] == ' ' :
        cap_modified = cap[0:-1]
    if cap[-2:] == '. ' :
        cap_modified = cap[0:-2]
    if cap[-3:] == '.  ' :
        cap_modified = cap[0:-3]
    
    cap_modified = cap_modified.replace(',', '')
    cap_modified = cap_modified.replace('-', '')
    cap_modified = cap_modified.replace(':', '')
    cap_modified = cap_modified.replace('.', '')
    cap_modified = cap_modified.replace('\n', '')
    cap_modified = cap_modified.replace('\t', '')
    cap_modified = cap_modified.replace("'", '')
    cap_modified = cap_modified.replace('"', '')
    cap_modified = cap_modified.replace(";", '')
    cap_modified = cap_modified.replace("  ", ' ')
    cap_modified = cap_modified.upper()
    # if cap_modified[-1] == ' ' :
    #     cap_modified = cap_modified[0:-1]
        
    return cap_modified
    

data_subset_images = {}
for counter_item, item in enumerate(data_subset):
    data_subset_images[item ['image_id']] = counter_item
    
# example_subset = data_subset[0] 
# example_subset_image_id = example_subset ['image_id']
# example_subset_caption = example_subset ['caption']
inds = []  
caps = []
caps_data = []
exceptions = []
exceptions_annotations = []
for example_data in data:
    example_data_image = example_data ['image']
    example_data_cap = example_data['caption']['text']
    example_data_image_id = img_path_to_id_all [example_data_image]
    if example_data_image_id in data_subset_images:
        ind = data_subset_images [example_data_image_id]
        cap = data_subset[ind]['caption']
        cap_data = example_data_cap
        #
        cap_modified = modify_cap (cap)
        #
        if cap_modified == cap_data:       
            inds.append(ind)
            caps.append(cap)
            caps_data.append(cap_data)
        if cap_modified[-30:] == cap_data[-30:] and cap_modified != cap_data:
            
            print(cap_modified)
            print(cap_data)
            exceptions.append(example_data ['image'])
            exceptions_annotations.append(cap_modified)

test = exceptions[7]
print(exceptions_annotations[7])
for example_data in data:
    example_data_image = example_data ['image']
    if example_data_image == test:
        print(example_data['caption']['text'])
#%%

r = []
i = []
for counter_item, item in enumerate(data_subset):
    data_subset_caption = [item ['caption']]
    if data_subset_caption[0] not in caps:
        r.append(data_subset_caption[0])
        print('----' + data_subset_caption[0] + '------')
        print('----' + modify_cap(data_subset_caption[0]) + '------')
        i.append(item['image_id'])
  
        
im_id = i[0]
cap_q = r[0]

        
# example_data = data[0]
# example_data_image = example_data ['image']
# example_data_cap = example_data['caption']['text']
# example_data_image_id = img_path_to_id_all [example_data_image]    