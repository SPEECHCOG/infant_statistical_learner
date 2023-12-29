import numpy
import os
import scipy
from utilsMSCOCO import read_data_from_path, get_all_cats
from utilsMSCOCO import sort_object, plot_dist_cats, change_labels
import json
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('/worktmp2/hxkhkh/current/Dcase/model/word2vec/GoogleNews-vectors-negative300.bin', binary=True)



def detec_nouns (input_words):
    tok = nltk.pos_tag(input_words)
    nouns = [n[0] for n in (tok) if n[1] =='NN' or n[1] =='NNS' ]
    noun_indexes = [counter for counter in range(len(tok)) if tok[counter][1] =='NN' or tok[counter][1] =='NNS' ]
    return nouns , noun_indexes

def get_unique_nouns (data_json):
    wavfile_nouns = []
    unique_nouns = {}
    for k in range(len(data_json['annotations'])):
        data_annFile_example = data_json['annotations'][k]
        caption_example = data_annFile_example['caption']
        
        words = word_tokenize(caption_example)
         
        nouns , noun_indexes = detec_nouns (words)
        for n in nouns:
            if n not in unique_nouns:
                unique_nouns[n] = 1
            else:
                unique_nouns[n] += 1
        wavfile_nouns.append(nouns)
    return unique_nouns, wavfile_nouns
    
###############################################################################    
dataDir='../data/coco_pyp/MSCOCO'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco , cats, cat_ids = read_data_from_path (dataDir, dataType)
cats_id_to_name, cats_id_to_supername, cats_name_to_id = get_all_cats (coco)
all_labels = change_labels(cats_id_to_name)
###############################################################################

def create_dict_images (data):
    dict_images = {}
    for item in data:
        i = item['image']
        cap = item['caption']['text']
        if i not in dict_images:
            dict_images[i] = []
            dict_images[i].append(cap)
        else:
            dict_images[i].append(cap)
    return dict_images

def find_label_for_noun (noun, all_labels, thresh):
    name = ''
    try:
        sim = [model.similarity(noun, label) for label in all_labels]
    except:
        sim = [0]
        pass
    if max(sim) >= thresh:
        name = all_labels [numpy.argmax(sim)]
    return name


def read_captions_from_dict_image(dict_images):
    dict_images_nouns = {}
    for key_image, caps in dict_images.items():
        dict_images_nouns [key_image] = {}
        dict_images_nouns [key_image]['caption'] = []
        dict_images_nouns [key_image]['words'] = []
        dict_images_nouns [key_image]['nouns'] = []
        dict_images_nouns [key_image]['names'] = []
        dict_images_nouns [key_image]['labels'] = {}
        for caption in caps:
            
            words = word_tokenize(caption) 
            words = [w.lower() for w in words]
            nouns , noun_indexes = detec_nouns(words)
            all_names = [find_label_for_noun (n, all_labels, thresh=0.65) for n in nouns]
            names = [n for n in all_names if n != '']
            dict_images_nouns [key_image]['caption'].append(caption)
            dict_images_nouns [key_image]['words'].append(words)
            dict_images_nouns [key_image]['nouns'].append(nouns)
            dict_images_nouns [key_image]['names'].append(names)
            for n in names:
                if n not in dict_images_nouns [key_image]['labels']:
                    dict_images_nouns [key_image]['labels'][n] = 1
                else: 
                    dict_images_nouns [key_image]['labels'][n] = dict_images_nouns [key_image]['labels'][n] + 1
    return dict_images_nouns

###############################################################################

###############################################################################
def read_captions_from_json(split):
    caption_json = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/MSCOCO/annotations/captions_" + split + "2014.json"
    with open(caption_json, 'r') as fp:
        data_json = json.load(fp)
    
    unique_nouns, wavfile_nouns = get_unique_nouns (data_json)
    #unique_nouns_sorted = sorted(unique_nouns.items(), key=lambda x:x[1], reverse=True)   
    return unique_nouns, wavfile_nouns
def sort_names (unique_nouns):
    names = list(unique_nouns.keys())
    values = list (unique_nouns.values())
    sorted_ind, names_sorted, values_sorted = sort_object (names, values)
    return sorted_ind, names_sorted, values_sorted
  
def plot_nouns_dist (names_sorted, values_sorted, save_path, split, limit):  
    save_name = save_path + split + '_nouns_' + str(limit)
    title = 'Number of namings for frequent unique names'
    plot_dist_cats (names_sorted[0:limit], values_sorted[0:limit] , save_name, title)

def plot_names_dist (names_sorted, values_sorted, save_path, split, limit, thresh):  
    save_name = save_path + split + '_names_' + str(limit) + '_' + str(thresh) 
    title = 'Number of namings for frequent unique names'
    plot_dist_cats (names_sorted[0:limit], values_sorted[0:limit] , save_name, title)
    
def find_objects_and_names (all_labels, names_sorted, limit, thresh, save_name):
    all_sims = [] 
    object_and_names_all = {}    
    for un in names_sorted [0:limit]:
        #print(un)
        try:
            sim = [model.similarity(un, label) for label in all_labels]
            all_sims.extend(sim)
        except:
            sim = 0
            pass
        if max(sim) >= thresh:
            l = all_labels [numpy.argmax(sim)]
            print(un + ' is very similar to ' + l)
            object_and_names_all [un] = l
    if save_name:       
        scipy.io.savemat(save_name, object_and_names_all)
    return all_sims, object_and_names_all



def replace_names_with_objects (all_labels, nouns_sorted, values_sorted, limit, thresh, save_name):
    unique_names_counts = {} 
    unique_names_nouns = {} 
    n_list = nouns_sorted[0:limit]
    v_list = values_sorted[0:limit]
    for counter, un in enumerate(n_list):
        #print(un)
        try:
            sim = [model.similarity(un, label) for label in all_labels]
        except:
            pass
        if max(sim) >= thresh:
            l = all_labels [numpy.argmax(sim)]
            print(un + ' is very similar to ' + l)
            if l in unique_names_counts:
                unique_names_counts [l] += v_list[counter]
                unique_names_nouns [l].append(un)
            else:
                unique_names_counts [l] = v_list[counter]
                unique_names_nouns [l] = []
                unique_names_nouns [l].append(un)
    if save_name:       
        scipy.io.savemat(save_name+ '_nouns_'+ str(thresh) + '.mat', unique_names_nouns)
        scipy.io.savemat(save_name + '_counts_'+ str(thresh) + '.mat', unique_names_counts)
    return unique_names_counts, unique_names_nouns

# ind_accepted = []    
# captions_rejected = []
# for counter_utterance, candidate_utterance  in enumerate(wavfile_nouns[0:200]):        
#     print(counter_utterance)
#     try:
#         model.most_similar(candidate_utterance)
#         ind_accepted.append(counter_utterance)
#     except:
#         captions_rejected.append(candidate_utterance)
#         print("An exception occurred in word:  " + str(candidate_utterance))
 
###############################################################################

split = 'train'
unique_nouns_train, wavfile_nouns_train = read_captions_from_json(split)
sorted_ind_train, nouns_sorted_train, values_sorted_train = sort_names (unique_nouns_train) 


split = 'val'
unique_nouns_validation, wavfile_nouns_validation = read_captions_from_json(split)
sorted_ind_validation, nouns_sorted_validation, values_sorted_validation = sort_names (unique_nouns_validation)

unique_nouns = {}
for key in unique_nouns_train:
    if key in unique_nouns_validation:
        unique_nouns[key] = unique_nouns_train[key] + unique_nouns_validation [key]
    else:
        unique_nouns[key] = unique_nouns_train[key] 
    
sorted_ind_all, nouns_sorted_all, values_sorted_all = sort_names (unique_nouns) 
############################################################################### plotting
save_path = '/worktmp2/hxkhkh/current/FaST/experiments/plots/vf/distributions/names/'
# limit = 80 
# 

# split = 'train'
# plot_nouns_dist (nouns_sorted_train, values_sorted_train, save_path , split , limit)

# split = 'val'
# plot_nouns_dist (nouns_sorted_validation, values_sorted_validation, save_path , split , limit)

# split = 'all'
# plot_nouns_dist (nouns_sorted_all, values_sorted_all, save_path , split , limit)

# limit = 1000
# thresh = 0.8
# path = '/worktmp2/hxkhkh/current/FaST/experiments/plots/vf/distributions/names/' 

# save_name = path + 'similar_objects_and_names_train.mat'
# all_sims, object_and_names_train = find_objects_and_names (all_labels, nouns_sorted_train, limit, thresh, save_name )

# save_name = path + 'similar_objects_and_names_validation.mat'
# all_sims, object_and_names_val = find_objects_and_names (all_labels, nouns_sorted_validation, limit, thresh, save_name )

# save_name = path + 'similar_objects_and_names_all.mat'
# all_sims, object_and_names_all = find_objects_and_names (all_labels, nouns_sorted_all, limit, thresh, save_name )

limit = 10000
thresh = 0.65
path = '/worktmp2/hxkhkh/current/FaST/experiments/plots/vf/distributions/names/'
save_name = path + 'names_all'
unique_names_counts, unique_names_nouns = replace_names_with_objects (all_labels, nouns_sorted_all, values_sorted_all, limit, thresh, save_name)
sorted_ind, names_sorted, values_sorted = sort_names (unique_names_counts)
split = 'all'
plot_names_dist (names_sorted, values_sorted, save_path , split , limit, thresh)
# from matplotlib import pyplot as plt
# plt.hist(all_sims , bins = 100) 

############################################################################### finding label distributons for images
data_root = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp"

audio_dataset_json_file = os.path.join(data_root, "SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json")
with open(audio_dataset_json_file, 'r') as fp:
    data_json = json.load(fp)
data_val = data_json['data']
example_cap = data_val [0]['caption']['text']

dict_images_val = create_dict_images (data_val)

audio_dataset_json_file = os.path.join(data_root, "SpokenCOCO/SpokenCOCO_train_unrolled_karpathy.json")
with open(audio_dataset_json_file, 'r') as fp:
    data_json = json.load(fp)
data_train = data_json['data']

dict_images_train = create_dict_images (data_train)

dict_images_names_train = read_captions_from_dict_image(dict_images_train)
dict_images_names_val = read_captions_from_dict_image(dict_images_val)

dict_images_names_all = {}
for key in dict_images_names_train:
    dict_images_names_all [key] = dict_images_names_train [key]   
for key in dict_images_names_val:
    dict_images_names_all [key] = dict_images_names_val [key]
    

save_path = '/worktmp2/hxkhkh/current/FaST/experiments/plots/vf/distributions/'
# save_name = save_path + 'dict_images_and_names_'
# scipy.io.savemat(save_name + 'all.mat', dict_images_names_all)
# scipy.io.savemat(save_name + 'train.mat', dict_images_names_train)
# scipy.io.savemat(save_name + 'val.mat', dict_images_names_val)

dict_labels_all = {}
for item in all_labels:
    dict_labels_all[item] = []
    
for key_image, value in  dict_images_names_all.items():
    labels_image  = value ['labels']
    for key_l, count_l in labels_image.items():
        dict_labels_all [key_l].append(count_l)
        
# save_name = save_path + 'dict_labels_'        
# scipy.io.savemat(save_name + 'all.mat', dict_labels_all) 
dict_labels_distributions = {}   
for key_l, list_count in dict_labels_all.items():
    dict_labels_distributions[key_l] = numpy.mean(list_count)
    
sorted_ind, items_sorted, values_sorted = sort_names (dict_labels_distributions)
split = 'all'
limit = 65
plot_nouns_dist (items_sorted, values_sorted, save_path , split , limit)

###############################################################################

filepath = '/worktmp2/hxkhkh/current/FaST/experiments/plots/vf/distributions/dict_images_and_names_all.mat'

d = scipy.io.loadmat(filepath)
