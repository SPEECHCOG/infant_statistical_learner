
import csv
import json
import os
import random
import soundfile as sf


# Function to read data from a TSV file
def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        data = [row[0] for row in reader]
    return data


def read_tsv_LS_Peng(file_path, max_keep, min_keep):
    
    hours = []
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    with open(file_path) as f:
        # Kh: I chaned this
        #root = f.readline().strip()
        # Kh
        f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
                
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                #kh: i changed this 
                #names.append(items[0])
                file_splits = items[0].split("/")[-4:]
                file_name = '/'.join(file_splits)
                         
                names.append(file_name)
                #kh
                
                inds.append(ind)
                sizes.append(sz)
                hours.append((sz/16000)/3600)
    tot = ind + 1           
    return names, inds, tot

# Function to read SC data from a JSON file
def read_wav_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data_json = json.load(json_file) 
    data = data_json['data']
    wavs = []
    for d in data:
        w = d['caption']['wav']
        wavs.append(w)
    return wavs

# Function to combine LS and SC data
# Note that due to the shuffling, the function should be run only once 
def combine_LS_SC (ls_names, sc_names):
    all_names = []
    for item in ls_names: 
        all_names.append(os.path.join('LS','wavs', item))
    for item in sc_names:
        all_names.append(os.path.join('coco_pyp','SpokenCOCO',item))
    random.shuffle(all_names)
    return all_names
            
        
# Function to write data to a TSV file
def write_list_to_tsv(file_path, data_list):
    with open(file_path, 'w', encoding='utf-8', newline='') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        for item in data_list:
            writer.writerow([item])

kh
#%%

# Combining LS train and COCO SSL train as "train.tsv"  

# File paths
tsv_file_path = '../../../FaST/datavf/libri_fn_root/Pengfiles/train.tsv'
json_file_path = '../../../FaST/datavf/coco/subsets/SpokenCOCO_train_SSL.json'


# File names
ls_names, inds, tot = read_tsv_LS_Peng(tsv_file_path, max_keep = 16000*80, min_keep = 32000)
sc_names = read_wav_json(json_file_path)



# Write SSL Libri to a new TSV file
combined_data_list = combine_LS_SC (ls_names, [])
out_file_path = '../../../FaST/datavf/libri_fn_root/train.tsv'
write_list_to_tsv(out_file_path, combined_data_list)

# Write SSL COCO to a new TSV file
combined_data_list = combine_LS_SC ([], sc_names)
out_file_path = '../../../FaST/datavf/ssl_COCO/train.tsv'
write_list_to_tsv(out_file_path, combined_data_list)

# Write combined data (LS + COCO) to a new TSV file
combined_data_list = combine_LS_SC (ls_names, sc_names)
out_file_path = '../../../FaST/datavf/ssl_root/train.tsv'
write_list_to_tsv(out_file_path, combined_data_list)

# Test the file content
test_data = read_tsv(out_file_path)

#%%

# Recreating LS "train.tsv" with the correct paths 

# File paths
tsv_file_path = '../../../FaST/datavf/libri_fn_root/Pengfiles/train.tsv'
out_file_path = '../../../FaST/datavf/libri_fn_root/train.tsv'

# File names
ls_names, inds, tot = read_tsv_LS_Peng(tsv_file_path, max_keep = 16000*80, min_keep = 32000)
sc_names = []

# Combine the LS and SC train data
combined_data_list = combine_LS_SC (ls_names, sc_names)

# Write combined data to a new TSV file
write_list_to_tsv(out_file_path, combined_data_list)
#%%

# Recreating LS "val.tsv" with the correct paths 

# File paths
tsv_file_path = '../../../FaST/datavf/libri_fn_root/Pengfiles/valid.tsv'
out_file_path = '../../../FaST/datavf/libri_fn_root/valid.tsv'

# File names
ls_names, inds, tot = read_tsv_LS_Peng(tsv_file_path, max_keep = 16000*80, min_keep = 32000)
sc_names = []

# Combine the LS and SC train data
combined_data_list = combine_LS_SC (ls_names, sc_names)

# Write combined data to a new TSV file
write_list_to_tsv(out_file_path, combined_data_list)
#%%
# Test the file content
out_file_path = '../../../FaST/datavf/ssl_root/trainRes.tsv'
test_data = read_tsv(out_file_path)

#%%
# measuring the total time 
# selecting SSL 6M and SSL Res (SSL_total - SSL_6M)

file_path = '../../../FaST/datavf/ssl_root/train.tsv'
SSL_data = read_tsv(file_path)

SSL_6M = []
SSL_res = []
sec6M = 1049* 3600

path_root = '../../../FaST/data'
seconds_orig = []
seconds_applied = []
seconds_6M = []
seconds_res = []
for wpath in SSL_data:
    
    path = os.path.join(path_root,wpath)
    x, sr = sf.read(path, dtype = 'float32')
    length_orig = len(x)
    time_orig = length_orig /sr
    seconds_orig.append(time_orig)
    
    # check if the total time is still less than 6 Months
    if (sum(seconds_orig)) <= sec6M :
        SSL_6M.append(wpath)
        seconds_6M.append(time_orig)
    else:
        SSL_res.append(wpath)
        seconds_res.append(time_orig)
        
    # measure applied seconds     
    if length_orig > sr * 8:
        seconds_applied.append(8)
    else:
        seconds_applied.append(time_orig)
    
hours = sum(seconds_orig)/3600
print(' ..... total time original is ....' + str(hours))

hours_applied = sum(seconds_applied)/3600
print(' ..... total time applied is ....' + str(hours_applied))

hours_6M = sum(seconds_6M)/3600
print(' ..... total time SSL 6M is ....' + str(hours_6M))

hours_res = sum(seconds_res)/3600
print(' ..... total time res is ....' + str(hours_res))

# Write SSL 6M data to a new TSV file
out_file_path = '../../../FaST/datavf/ssl_root/train6M.tsv'
write_list_to_tsv(out_file_path, SSL_6M )

out_file_path = '../../../FaST/datavf/ssl_root/trainRes.tsv'
write_list_to_tsv(out_file_path, SSL_res )

#%%
# measuring the total time 
# selecting SSL sub1, sub2, sub3

import soundfile as sf
import os 
file_path = '../../../FaST/datavf/sslRes_root/train.tsv'
SSL_res = read_tsv(file_path)

file_path = '../../../FaST/datavf/ssl6M_root/train.tsv'
SSL_6M = read_tsv(file_path)


sec2M = 5.83 * 60 * 3600
sec4M = 5.83 * 120 * 3600
sec6M = 5.83 * 180 * 3600

path_root = '../../../FaST/data'

SSL_sub1 = []
SSL_sub2 = []
SSL_sub3 = []


seconds_orig = []
seconds_sub1 = []
seconds_sub2 = []
seconds_sub3 = []

for wpath in SSL_res:
    
    path = os.path.join(path_root,wpath)
    x, sr = sf.read(path, dtype = 'float32')
    length_orig = len(x)
    time_orig = length_orig /sr
    seconds_orig.append(time_orig)
    
    # check if the total time is still less than 6 Months
    if (sum(seconds_orig)) <= sec2M :
        
        SSL_sub1.append(wpath)
        seconds_sub1.append(time_orig)
        SSL_sub2.append(wpath)
        seconds_sub2.append(time_orig)
        SSL_sub3.append(wpath)
        seconds_sub3.append(time_orig)
    elif (sum(seconds_orig)) <= sec4M :
        SSL_sub2.append(wpath)
        seconds_sub2.append(time_orig)
        SSL_sub3.append(wpath)
        seconds_sub3.append(time_orig)
    elif (sum(seconds_orig)) <= sec6M :
        SSL_sub3.append(wpath)
        seconds_sub3.append(time_orig)
    
#%%
# Fill the rest with SSL 6 M

for wpath in SSL_6M:
    
    path = os.path.join(path_root,wpath)
    x, sr = sf.read(path, dtype = 'float32')
    length_orig = len(x)
    time_orig = length_orig /sr
    seconds_orig.append(time_orig)
    
    # check if the total time is still less than 6 Months
    
    # if (sum(seconds_orig)) <= sec2M :
    #     SSL_sub1.append(wpath)
    #     seconds_sub1.append(time_orig)
    if (sum(seconds_orig)) <= sec4M :
        SSL_sub2.append(wpath)
        seconds_sub2.append(time_orig)
        SSL_sub3.append(wpath)
        seconds_sub3.append(time_orig)
    elif (sum(seconds_orig)) <= sec6M :
        SSL_sub3.append(wpath)
        seconds_sub3.append(time_orig)
    else:
        break
#%%        
# Check the total time

h = sum(seconds_sub1)/3600
print(' ..... total time sub1 ....' + str(h))

h = sum(seconds_sub2)/3600
print(' ..... total time sub2 ....' + str(h))

h = sum(seconds_sub3)/3600
print(' ..... total time sub3 ....' + str(h))

# Write SSL subsets to a new TSV file
out_file_path = '../../../FaST/datavf/ssl_sub1_root/train.tsv'
write_list_to_tsv(out_file_path, SSL_sub1 )

out_file_path = '../../../FaST/datavf/ssl_sub2_root/train.tsv'
write_list_to_tsv(out_file_path, SSL_sub2 )

out_file_path = '../../../FaST/datavf/ssl_sub3_root/train.tsv'
write_list_to_tsv(out_file_path, SSL_sub3 )

#%%

# separate SC and LS in SSL6M

save_path_SC = "/worktmp2/hxkhkh/current/FaST/datavf/ssl6M_SC/train.tsv"
save_path_LS = "/worktmp2/hxkhkh/current/FaST/datavf/ssl6M_LS/train.tsv"
path_mix = "/worktmp2/hxkhkh/current/FaST/datavf/ssl6M_root"

data_mix = read_tsv(os.path.join(path_mix, "train.tsv"))

data_SC = []
data_LS = []

for d in data_mix:
    if d[0] == "c":
        data_SC.append(d)
    else:
        data_LS.append(d)

write_list_to_tsv(save_path_SC, data_SC)
write_list_to_tsv(save_path_LS, data_LS )