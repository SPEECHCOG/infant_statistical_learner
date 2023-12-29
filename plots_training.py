import os
import json
import numpy as np
from matplotlib import pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

# global variables
root = "/worktmp2/hxkhkh/current/"
path_abx = os.path.join(root, 'ZeroSpeech/output/AC/DINO/exp6M/')
path_lex = os.path.join(root, "lextest/output/CDI/DINO/exp6M")
path_sem = os.path.join(root, "semtest/S/DINO/exp6M")
path_save = "/worktmp2/hxkhkh/current/FaST/papers/vf/material/"
dtype = 'CDI'
mtype = 'DINO'



names = ["0", "6 ", "8 ", "10 ", "12 ", "10(u)"]


    
    
#%% Recall    
# manually enter numbers for recall@10 [ssl, r1, r2, r0, r3]


# S1: 6M: Audio R@10 0.009 Image R@10 0.007     FB: Audio R@10 0.011 Image R@10 0.007
# S2: 6M: Audio R@10 0.061 Image R@10 0.052    FB: Audio R@10 0.051 Image R@10 0.044
# S3: 6M: Audio R@10 0.117 Image R@10 0.098     FB: Audio R@10 0.116 Image R@10 0.092
#................
# S0: 6M: Audio R@10 0.064 Image R@10 0.050     FB: Audio R@10 0.055 Image R@10 0.050


x_recall = ['baseline','exp6M', 'expS1', 'expS2', 'expS3' , 'expS0']
r_image = [0.2, 0.2, 0.7, 5.2 , 9.8 , 5.0 ]
r_speech = [0.2, 0.2, 0.9, 6.1, 11.7 , 6.4]
results_recall = [r_image, r_speech]

#%%
path_source = '/worktmp2/hxkhkh/current/FaST/experiments/vfsubsets/exp6M/'

path_S0 = 'expS0/'
path_S1 = 'expS1/'
path_S2 = 'expS2/'
path_S3 = 'expS3/'
path_ssl = "/worktmp2/hxkhkh/current/FaST/experiments/vfls/expnewl15"

paths = [path_S1, path_S2, path_S3, path_S0 ]
steps = [20, 40, 60, 40]

#%%

def find_single_recall (event, n):
    recall = pd.DataFrame(event.Scalars('acc_r10'))   
    x_recall = recall['step'].to_list()#[i/n for i in recall['step']]
    y_recall =  recall['value'].to_list()    
    return x_recall, y_recall


def find_single_vgsloss (event, interval):
    vgsloss = pd.DataFrame(event.Scalars('coarse_matching_loss')) #coarse_matching_loss
    x_vgsloss = vgsloss['step'] [::interval]  #[ i/n for i in vgsloss['step']][::interval] 
    y_vgsloss = vgsloss['value'][::interval]
    y_vgsloss_list = y_vgsloss.to_list()
    return x_vgsloss, y_vgsloss_list

def find_single_caploss (event, interval):
    vgsloss = pd.DataFrame(event.Scalars('caption_w2v2_loss'))
    x_vgsloss = vgsloss['step'] [::interval]  #[ i/n for i in vgsloss['step']][::interval] 
    y_vgsloss = vgsloss['value'][::interval]
    y_vgsloss_list = y_vgsloss.to_list()
    return x_vgsloss, y_vgsloss_list
 
kh
#%%
# W2V2 loss

# event =  EventAccumulator(path_ssl)
# event.Reload()
# caploss = pd.DataFrame(event.Scalars('libri_w2v2_loss'))
# x_6M = caploss['step'] [::100]  #[ i/n for i in vgsloss['step']][::interval] 
# l_6M = caploss['value'][::100]
# x_6M = x_6M.to_list()
# l_6M = l_6M.to_list()
# np.save('x_6M',x_6M )
# np.save('l_6M',l_6M)

# capvloss = pd.DataFrame(event.Scalars('vloss_cap'))
# xv_6M = capvloss['step'] [::43]  #[ i/n for i in vgsloss['step']][::interval] 
# lv_6M = capvloss['value'][::43]
# xv_6M = xv_6M.to_list()
# lv_6M = lv_6M.to_list()

# np.save('xv_6M',xv_6M )
# np.save('lv_6M',lv_6M )

xv_6M = np.load('xv_6M.npy')
lv_6M = np.load('lv_6M.npy')

x_6M = np.load('x_6M.npy')
l_6M = np.load('l_6M.npy')

plt.plot(l_6M)

xs = [x_6M]
cs = [l_6M]
paths = [ path_S1, path_S2, path_S3, path_S0 ]
intervals = [20,39,59,41]
for counter, p in enumerate(paths):
    event =  EventAccumulator(os.path.join(path_source, p))
    event.Reload()
    x, l = find_single_caploss(event, intervals[counter]) 
    xs.append(x)
    cs.append(l)

#%%
# wav2vec_fig

plt.figure(figsize =(12,12))

f_leg = 28
f_ticks = 26
f_ylabel = 34
lw = 5
plt.subplot(1,1,1)
plt.plot(cs[0], label='6 months (training loss)', linewidth = lw, color = 'gray')
plt.plot(lv_6M, label='6 months (validation loss)', linewidth = lw, color = 'gray', linestyle='dotted')
# plt.plot(cs[1], label='8 months', linewidth = lw)
# plt.plot(cs[2], label='10 months', linewidth = lw)
# plt.plot(cs[3], label='12 months', linewidth = lw)
# plt.plot(cs[4], label='10 months uniform', linewidth = lw)
plt.grid()
plt.legend(fontsize = f_leg)
plt.ylabel('loss_AUD', fontsize = f_ylabel)
plt.xlabel('epochs', fontsize = f_ylabel)
plt.xticks(fontsize = f_ticks)
plt.yticks(fontsize = f_ticks)
#plt.xticks([0,1,2,3,4,5,6,7,8,9,10], [str(item) for item in np.arange(0, 110, 10)])
plt.savefig(path_save +  'results_training_w2v2_6M_val.pdf' ,  format = 'pdf', bbox_inches='tight' )
#%%
# wav2vec_fig

plt.figure(figsize =(24,12))

f_leg = 28
f_ticks = 26
f_ylabel = 34
lw = 5
plt.subplot(1,2,1)
plt.plot(cs[0], label='6 months (training loss)', linewidth = lw, color = 'gray')
plt.plot(lv_6M, label='6 months (validation loss)', linewidth = lw, color = 'gray', linestyle='dotted')
plt.grid()
plt.legend(fontsize = f_leg)
plt.ylabel('loss_AUD', fontsize = f_ylabel)
plt.xlabel('epochs', fontsize = f_ylabel)
plt.subplot(1,2,2)

plt.plot(cs[1], label='8 months', linewidth = lw)
plt.plot(cs[2], label='10 months', linewidth = lw)
plt.plot(cs[3], label='12 months', linewidth = lw)
plt.plot(cs[4], label='10 months uniform', linewidth = lw)
plt.grid()
plt.legend(fontsize = f_leg)

plt.xlabel('epochs', fontsize = f_ylabel)
plt.xticks(fontsize = f_ticks)
plt.yticks(fontsize = f_ticks)
#plt.xticks([0,1,2,3,4,5,6,7,8,9,10], [str(item) for item in np.arange(0, 110, 10)])
plt.savefig(path_save +  'results_training_w2v2_6M_val_dual.pdf' ,  format = 'pdf', bbox_inches='tight' )



#%%
# Recall

xrs = []
rs = []
for counter, p in enumerate(paths):
    event =  EventAccumulator(os.path.join(path_source, p))
    event.Reload()
    x, r = find_single_recall(event, steps[counter]) 
    xrs.append(x)
    rs.append(r)
rs[1].insert(-1, max(rs[1]))

# VG loss

intervals = [20,39,59,43]
xls = []
ls = []
for counter, p in enumerate(paths):
    event =  EventAccumulator(os.path.join(path_source, p))
    event.Reload()
    x, l = find_single_vgsloss(event, intervals[counter]) 
    xls.append(x)
    ls.append(l)


   
#%%

plt.figure(figsize =(24,12))

f_leg = 28
f_ticks = 26
f_ylabel = 34
lw = 5
plt.subplot(1,2,2)

plt.plot(rs[0], label='8 months', linewidth = lw)
plt.plot(rs[1], label='10 months', linewidth = lw)
plt.plot(rs[2], label='12 months', linewidth = lw)
plt.plot(rs[3], label='10 months uniform', linewidth = lw)
plt.grid()
plt.ylabel('cross-modal retrieval (%)', fontsize = f_ylabel)
plt.xlabel('epochs', fontsize = f_ylabel)
plt.xticks([0,1,2,3,4,5,6,7,8,9,10], [str(item) for item in np.arange(0, 110, 10)],fontsize = f_ticks)
plt.yticks(fontsize = f_ticks)
plt.savefig(path_save +  'results_plot_recall.pdf' ,  format = 'pdf', bbox_inches='tight' )


plt.subplot(1,2,1)
plt.plot(ls[0], label='8 months', linewidth = lw)
plt.plot(ls[1], label='10 months', linewidth = lw)
plt.plot(ls[2], label='12 months', linewidth = lw)
plt.plot(ls[3], label='10 months uniform', linewidth = lw)
plt.grid()
plt.legend(fontsize = f_leg)
plt.ylabel('loss_AV', fontsize = f_ylabel)
plt.xlabel('epochs', fontsize = f_ylabel)
plt.xticks(fontsize = f_ticks)
plt.yticks(fontsize = f_ticks)
#plt.xticks([0,1,2,3,4,5,6,7,8,9,10], [str(item) for item in np.arange(0, 110, 10)])
plt.savefig(path_save +  'results_training.pdf' ,  format = 'pdf', bbox_inches='tight' )   

#%%


    
# rs = ys
# plt.figure()
# plt.plot(rs[0], label='8 months')
# plt.plot(rs[1], label='10 months')
# plt.plot(rs[2], label='12 months')
# plt.plot(rs[3], label='10 months uniform')
# plt.grid()
# plt.legend()
# plt.ylabel('W2V2 loss', fontsize = 12)
# plt.xlabel('epochs', fontsize = 12)
# #plt.xticks([0,1,2,3,4,5,6,7,8,9,10], [str(item) for item in np.arange(0, 110, 10)])
# plt.savefig(path_save +  'results_plot_cap_loss.pdf' ,  format = 'pdf', bbox_inches='tight' )  

