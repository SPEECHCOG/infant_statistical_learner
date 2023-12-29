dict_rws = {}


dict_rws ['airplane'] = 0.06
dict_rws ['apple'] = 0.84
dict_rws ['ball'] = 0.42
dict_rws ['balloon'] = 0.18 
dict_rws ['banana'] = 1.08
dict_rws ['bed'] = 0.30
dict_rws ['bee'] = 0.36
dict_rws ['bib'] = 1.08
dict_rws ['bicycle'] = 0.36
dict_rws ['bird'] = 0.12
dict_rws ['blanket'] = 0.06
dict_rws ['block'] = 0.06
dict_rws ['book'] = 2.33
dict_rws ['boots'] = 0.12
dict_rws ['bottle'] = 0.48
dict_rws ['bowl'] = 1.08
dict_rws ['box'] = 0.18
dict_rws ['bread'] = 1.20
dict_rws ['bug'] = 0.06
dict_rws ['bunny'] = 0.12
dict_rws ['bus'] = 0.06
dict_rws ['butterfly'] = 0.06
dict_rws ['button'] = 0.72
dict_rws ['candy'] = 0.06
dict_rws ['car'] = 0.36
dict_rws ['carcarrots'] = 1.43
dict_rws ['cat'] = 0.42
dict_rws ['cereal'] = 0.42
dict_rws ['chair'] = 0.84
dict_rws ['cheerios'] = 0.18
dict_rws ['cheese'] = 2.51
dict_rws ['chicken'] = 0.66
dict_rws ['clock'] = 0.06
dict_rws ['coat'] = 0.06
dict_rws ['cookie'] = 0.24
dict_rws ['couch'] = 0.30
dict_rws ['cracker'] = 0.96
dict_rws ['cup'] = 0.66
dict_rws ['diaper'] = 0.42
dict_rws ['dog'] =1.49
dict_rws ['door'] = 0.48
dict_rws ['drawer'] = 0.24
dict_rws ['drink'] = 0.66
dict_rws ['duck'] = 0.18
dict_rws ['egg'] = 3.17
dict_rws ['fish'] = 0.54
dict_rws ['fork'] = 0.66
dict_rws ['frog'] = 0.06
dict_rws ['glass'] = 0.18
dict_rws ['glasses'] = 0.12
dict_rws ['horse'] = 0.06
dict_rws ['house'] = 1.31
dict_rws ['icecream'] = 0.12
dict_rws ['jacket'] = 0.18
dict_rws ['juice'] = 1.85
dict_rws ['kitty'] = 0.18
dict_rws ['lion'] = 0.06
dict_rws ['medicine'] = 0.12
dict_rws ['milk'] = 1.55
dict_rws ['moon'] = 0.48
dict_rws ['orange'] = 0.54
dict_rws ['oven'] = 0.12
dict_rws ['pants'] = 0.18
dict_rws ['paper'] = 2.45
dict_rws ['peas'] = 0.54
dict_rws ['pen'] = 0.30
dict_rws ['pillow'] = 0.12
dict_rws ['pizza'] = 1.02
dict_rws ['plant'] = 0.12
dict_rws ['plate'] = 1.49
dict_rws ['potty'] = 0.24
dict_rws ['shirt'] = 1.20
dict_rws ['shoe'] = 0.54
dict_rws ['sink'] = 0.24
dict_rws ['sky'] = 0.06
dict_rws ['soap'] = 0.06
dict_rws ['sock'] = 0.30
dict_rws ['spoon'] = 1.55
dict_rws ['sun'] = 0.06
dict_rws ['swing'] = 0.06
dict_rws ['table'] = 1.55
dict_rws ['telephone'] = 1.02
dict_rws ['towel'] = 0.06
dict_rws ['toy'] = 1.08
dict_rws ['tree'] = 0.30
dict_rws ['truck'] = 0.06
dict_rws ['tv'] = 0.12
dict_rws ['water'] = 2.15
dict_rws ['window'] = 0.12


dict_rws_sorted = sorted (dict_rws.items(), key=lambda x:x[1], reverse=True)
rws_counts_sorted = []
for item in dict_rws_sorted:
    rws_counts_sorted.append(item[1])
    
from scipy.io import savemat
save_path =   '/worktmp2/hxkhkh/current/FaST/plots/vf/distributions/new/' 
 
save_name = save_path + 'rws_counts_sorted.mat'
mdict = {'data': rws_counts_sorted}
savemat(save_name, mdict)


from scipy.io import loadmat
file_name = save_path + 'rws_counts_sorted.mat'
test_rws_sorted = loadmat(file_name, variable_names = 'data')
data = test_rws_sorted['data']
