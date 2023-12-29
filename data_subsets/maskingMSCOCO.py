import numpy
import cv2
import os
from matplotlib import pyplot as plt
from utilsMSCOCO import read_data_from_path, get_all_cats, get_all_image_ids
import copy
# global variables 

# from pycocotools.coco import COCO
# import pylab
# pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# dataDir='../data/coco_pyp/MSCOCO'
# dataType='val2014'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# anncaptionFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
# coco=COCO(annFile)
# coco_caps=COCO(anncaptionFile)
# cats = coco.loadCats(coco.getCatIds())
# cats_id = [item['id'] for item in cats]
# cats_names = [item['name']for item in cats]


# def compute_GT_mask (label, imID, imH, imW): # , res_target_w , res_target_h):      
#     anncatind = cats_names.index(label)
#     anncatID = cats_id[anncatind]
#     annIds_imgs = coco.getAnnIds( imgIds=imID,catIds=anncatID, iscrowd=False)
#     anns = coco.loadAnns(annIds_imgs)
#     mask_annitem = numpy.zeros([imH,imW])
#     for item in anns: # constructing true mask by ading all mask items
#         mask_temp = coco.annToMask(item )
#         mask_annitem = mask_annitem + mask_temp
#     #mask_annitem = cv2.resize(mask_annitem, (res_target_w,res_target_h))       
#     return mask_annitem, anncatind

# imID = 42
# label_id = cats_id [0]
# label = cats_names [0]
# img = coco.loadImgs(imID)[0]
# imID = img['id']
# imW = img['width']
# imH = img['height']
# mask_GT, label_id = compute_GT_mask (label, imID, imH, imW)
# plt.imshow(mask_GT)

###############################################################################
                ############# masking images #############
###############################################################################

dataDir='../data/coco_pyp/MSCOCO'
dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco , cats, cat_ids = read_data_from_path (dataDir, dataType)
cats_id_to_name, cats_id_to_supername, cats_name_to_id = get_all_cats (coco)

img_ids = get_all_image_ids (coco)

image_id = img_ids [0]
img = coco.loadImgs(image_id)[0]
h = img ['height']
w = img ['width']
name = img ['file_name']
imPath = os.path.join(dataDir, dataType,name )
image = cv2.imread(imPath)
annId_img = coco.getAnnIds( imgIds=image_id, iscrowd=False) 
anns_image = coco.loadAnns(annId_img)
mask_annitem = numpy.zeros([h,w])
for item in anns_image : # constructing true mask by ading all mask items
    mask_temp = coco.annToMask(item)
    mask_annitem = mask_annitem + mask_temp

plt.imshow(mask_temp)
plt.imshow(image)
print (len(annId_img))
#%%
###############################################################################

masked_image = copy.deepcopy (image)
masked_image[:,:,0] = image[:,:,0] * mask_annitem
masked_image[:,:,1] = image[:,:,1] * mask_annitem
masked_image[:,:,2] = image[:,:,2] * mask_annitem

plt.imshow(masked_image)


blured_image = cv2.blur(image,(30,30),0)
plt.imshow(blured_image)

masked_image_blur = copy.deepcopy (blured_image)
masked_image_blur[:,:,0] = blured_image[:,:,0] * (1- mask_annitem)
masked_image_blur[:,:,1] = blured_image[:,:,1] * (1- mask_annitem)
masked_image_blur[:,:,2] = blured_image[:,:,2] * (1- mask_annitem)


plt.imshow(masked_image_blur )
blurmasked_image = masked_image_blur + masked_image
plt.imshow(blurmasked_image)


savePath = '/worktmp2/hxkhkh/current/FaST/experiments/plots/vf/masked_images/'
cv2.imwrite(savePath +  name , image)
cv2.imwrite(savePath + 'masked_' + name , masked_image)
cv2.imwrite(savePath + 'blurmasked30_' + name , blurmasked_image)
