import json 
import pandas as pd
import scipy
import numpy
#%% reading data

p_rws = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/rws_counts_sorted.mat"
path_freq_counts = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/dict_words_selected_counts.json"
path_label_word = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/dict_words_selected.json"
p = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/subsets/nsub3_meta.json"
pu = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/subsets/nsub0_meta.json"

with open(path_freq_counts, "r") as file:
    d_c = json.load(file)
    
with open(path_label_word, "r") as file:
    d_w = json.load(file)
    
with open(p, "r") as file:
    d_meta = json.load(file)
    
with open(pu, "r") as file:
    d_meta_u = json.load(file)

d_counts = sorted(d_c.items(), key=lambda x:x[1], reverse=True)
#d_frq =  d_meta ['object_freq']
d_area =  d_meta ['object_areas']
d_area_u =  d_meta_u ['object_areas']
# for key in d_area:
#     print(d_area[key])
#     print(d_area_u[key])
#     print('...............')
    
    
d_rws = scipy.io.loadmat(p_rws)['data'][0]
rws = numpy.sort(d_rws)[::-1]
d_frq = {}
fr_all = []
for counter, value in enumerate(d_counts):
    l = value[0]
    f = round ( (rws[counter]*0.5 * (56.1/60)) , 3)
    d_frq[l] = f
    fr_all.append(f)
fr_uniform = numpy.mean(fr_all)

#%% Semtest scores

path_semtest_results = "/worktmp2/hxkhkh/current/semtest/results/Semtest_categories.json"

with open(path_semtest_results, 'r') as file:
    d_results = json.load(file) ['DINO']
#%% Table
table = {}
count = 1
output_l = {}
output_w = {}
output_ws = {}
output_f = {}
output_a = {}

output_s8 = {}
output_s10 = {}
output_s12 = {}
output_sU = {}

for item in d_counts:
    label = item[0]
    word = d_w [label]
    freq = d_frq [label]
    area = round (d_area [label] * 100 , 2)
    
    table [count] = {}
    table [count]['label'] = label
    table [count]['word'] = word
    table [count]['freq'] = freq
    table [count]['area'] = area    
    table[count]['Semtest-8M'] = d_results ['8 months'][label]
    table[count]['Semtest-10M'] = d_results ['10 months'][label]
    table[count]['Semtest-12M'] = d_results ['12 months'][label]
    table[count]['uniform'] = d_results ['uniform'][label]
     
    output_l[count] = label
    output_w[count] = word[0] 
    names = word[0]
    if len(word) > 1:
        for name in word[1:]:
            names = names + ', ' + name 

    output_ws[count] = names

    # output_ws[count] = pd.Series(word)
    output_f[count] = freq
    output_a[count] = area
    output_s8[count] = d_results ['8 months'][label]
    output_s10[count] = d_results ['10 months'][label]
    output_s12[count] = d_results ['12 months'][label]
    output_sU[count] = d_results ['uniform'][label]

    count += 1
#%% saving the results
path = "/worktmp2/hxkhkh/current/FaST/papers/vf/material/datatable/"
data = [output_l, output_w,  output_f, output_a, output_s8, output_s10, output_s12, output_sU]
data_names = ['output_l', 'output_w' ,'output_f', 'output_a', 'output_s8', 'output_s10', 'output_s12', 'output_sU']
for counter, item in enumerate(data):
    df = pd.DataFrame(data=item, index=[0])
    df = (df.T)
    df.to_excel(path + data_names[counter] + '.xlsx')

item = output_ws
df = pd.DataFrame(data=item, index=[0])
df = (df.T)
df.to_excel(path + 'output_ws' + '.xlsx')
    
import pandas as pd
df = pd.read_excel("/worktmp2/hxkhkh/current/FaST/papers/vf/material/datatable/tableDataV1.xlsx" , index_col=None)
df.to_latex("/worktmp2/hxkhkh/current/FaST/papers/vf/material/datatable/table1.tex", escape=False)

df = pd.read_excel("/worktmp2/hxkhkh/current/FaST/papers/vf/material/datatable/tableDataV2.xlsx", index_col=None)
df.to_latex("/worktmp2/hxkhkh/current/FaST/papers/vf/material/datatable/table2.tex", escape=False)