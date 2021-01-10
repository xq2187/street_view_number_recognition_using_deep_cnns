import tensorflow
import os
import tarfile
import cv2
import h5py
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageOps

def get_label_bound(index, data, debug=False):
    attrs = {}
    item = data['digitStruct']['bbox'][index].item()
    key_list = ['label', 'left', 'top', 'width', 'height']
    for key in key_list:
        attr = data[item][key]
        values = []
        for i in range(len(attr)):
            if len(attr) > 1:
                values.append(data[attr.value[i].item()].value[0][0])
            else:
                values = [attr.value[0][0]]
        attrs[key] = values
    return attrs

def get_cropped_bound(data_path , index, debug=False):
    #get_img_bbox(data_path ="/data/test/digitStruct.mat",index)
    file_data = h5py.File(data_path, 'r')
    attrs = get_label_bound(index, file_data)
    if debug:
        print(attrs)
    length = len(attrs['label'])
    min_left = min(attrs['left'])
    min_top = min(attrs['top'])
    max_right = max([attrs['width'][i] + attrs['left'][i] for i in range(min(len(attrs['width']),len(attrs['left'])))])
    max_bottom = max([attrs['height'][i] + attrs['top'][i] for i in range(min(len(attrs['height']),len(attrs['top'])))])
    if debug:
        print('min_left,min_top,max_right,max_bottom: ',min_left,min_top,max_right,max_bottom)
    mid_x = (min_left + max_right) / 2.0
    mid_y = (min_top + max_bottom) / 2.0
    #cropped the image to a
    max_side = max(max_right - min_left, max_bottom - min_top)

    bbox_left, bbox_top, bbox_width, bbox_height = (mid_x - max_side / 2.0,
                                                    mid_y - max_side / 2.0,
                                                    max_side,
                                                    max_side)

    cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),
                                                                int(round(bbox_top - 0.15 * bbox_height)),
                                                                int(round(bbox_width * 1.3)),
                                                                int(round(bbox_height * 1.3)))
    if debug:
        print('cropped_left, cropped_top, cropped_width, cropped_height:',cropped_left, cropped_top, cropped_width, cropped_height)
    if debug:
        print('min_left,min_top,max_right,max_bottom',min_left,min_top,max_right,max_bottom)

    return (cropped_left, cropped_top, cropped_width, cropped_height)

def main(path_img, path_mat, path_cropped, debug=False):
    mat_file = h5py.File(os.getcwd()+path_mat,'r')
    for i in range(mat_file['/digitStruct/bbox'].shape[0]):
        index = i
        img_name = f"{i+1}.png"
        img = Image.open(os.getcwd() + f"{path_img}/{img_name}")
        #print(index)
        left,top,width,height = get_cropped_bound(os.getcwd()+path_mat,index,debug=debug)
        #currentAxis = plt.gca()
        #currentAxis.imshow(img)
        img_cropped = img.crop((left, top, left + width, top + height))
        #img1.show()
        img_cropped = img_cropped.resize([64,64])
        img_cropped.save(os.getcwd() + f"{path_cropped}/{i+1}.png")

