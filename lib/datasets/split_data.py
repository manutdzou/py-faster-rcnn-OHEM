# --------------------------------------------------------
# Copyright (c) 2016 RICOH
# Written by Zou Jinyi
# --------------------------------------------------------
import os
import sys
import cv2
import scipy.sparse

import numpy as np
import xml.dom.minidom as minidom
import cPickle

def load_image_set_index(data_path,image_set):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join(data_path, 'ImageSets', 'Main',
                                  image_set + '.txt')
    assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]
    return image_index

def load_pascal_annotation(data_path,index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    classes = ('__background__', # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(classes, xrange(len(classes))))

    filename = os.path.join(data_path, 'Annotations', index + '.xml')
    # print 'Loading: {}'.format(filename)
    def get_data_from_tag(node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data

    with open(filename) as f:
        data = minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')
    sizes = data.getElementsByTagName('size')
    if not 0:
        # Exclude the samples labeled as difficult
        non_diff_objs = [obj for obj in objs
                         if int(get_data_from_tag(obj, 'difficult')) == 0]
        if len(non_diff_objs) != len(objs):
            print 'Removed {} difficult objects' \
                .format(len(objs) - len(non_diff_objs))
        objs = non_diff_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, len(classes)), dtype=np.float32)

    for ind,size in enumerate(sizes):
        width=get_data_from_tag(size, 'width')
        height=get_data_from_tag(size, 'height')
        image_size=[int(width),int(height)]

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        # Make pixel indexes 0-based
        x1 = float(get_data_from_tag(obj, 'xmin')) - 1
        y1 = float(get_data_from_tag(obj, 'ymin')) - 1
        x2 = float(get_data_from_tag(obj, 'xmax')) - 1
        y2 = float(get_data_from_tag(obj, 'ymax')) - 1
        cls = class_to_ind[
                str(get_data_from_tag(obj, "name")).lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
    return {'boxes' : boxes,
            'gt_classes': gt_classes,
            'flipped' : False,
            'gt_overlaps' : overlaps,
            'size':image_size}

def append_flipped_images(num_images, gt_roidb):
    widths = [gt_roidb[i]['size'][0]
              for i in xrange(num_images)]
    for i in xrange(num_images):
        boxes = gt_roidb[i]['boxes'].copy()
        image_size=gt_roidb[i]['size']
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = widths[i] - oldx2 - 1
        boxes[:, 2] = widths[i] - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        entry = {'boxes' : boxes,
                 'gt_classes' : gt_roidb[i]['gt_classes'],
                 'flipped' : True,
                 'size':image_size}
        gt_roidb.append(entry)
    return gt_roidb

def image_path_at(i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return image_path_from_index(image_index[i])

def image_path_from_index(index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(data_path, 'JPEGImages',
                              index + image_ext)
    assert os.path.exists(image_path), \
           'Path does not exist: {}'.format(image_path)
    return image_path

def scale_and_ratio(min_size,max_size,image_size):
    image_size_min=min(image_size)
    image_size_max=max(image_size)
    im_scale = float(min_size) / float(image_size_min)
    if np.round(im_scale * image_size_max) > max_size:
        im_scale = float(max_size) / float(image_size_max)
    ratio=float(image_size[0])/float(image_size[1])
    return im_scale, ratio

#=======================================================================================#

if __name__ == '__main__':
    data_path='/home/bsl/py-faster-rcnn-master/data/VOCdevkit2007/VOC2007'
    image_set='trainval'
    image_ext = '.jpg'
    image_index=load_image_set_index(data_path,image_set)
    gt_roidbs = [load_pascal_annotation(data_path,index) for index in image_index]
    num_images=len(gt_roidbs)
    min_size=600
    max_size=1000
    gt_roidb=[]
    f=open('my_train.txt','w')
    for i in range(len(gt_roidbs)):
        image_size = gt_roidbs[i]['size']
        im_scale,ratio = scale_and_ratio(min_size,max_size,image_size)
        projection_gt = np.array(gt_roidbs[i]['boxes'])*im_scale
        width=projection_gt[:,2]-projection_gt[:,0]
        height=projection_gt[:,3]-projection_gt[:,1]
        index_scale=np.where((width>130) & (height>130))#130 for conv4-3, >130 for conv5-3
        if len(index_scale[0])>0:
            f.write(image_index[i]+'\n')
            boxes = gt_roidbs[i]['boxes'][index_scale]
            gt = gt_roidbs[i]['gt_classes'][index_scale]
            overlaps = gt_roidbs[i]['gt_overlaps'][index_scale]
            overlaps = scipy.sparse.csr_matrix(overlaps)
            entry = {'boxes' : boxes,
             'gt_classes' : gt,
             'flipped' : False,
             'gt_overlaps':overlaps}
            gt_roidb.append(entry)
    f.close()
    output_dir='/home/bsl/py-faster-rcnn-master/data/cache'
    gt_file = os.path.join(output_dir, 'gt_small.pkl')
    with open(gt_file, 'wb') as f:
        cPickle.dump(gt_roidb, f, cPickle.HIGHEST_PROTOCOL)


    
