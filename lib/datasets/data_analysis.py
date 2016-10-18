# --------------------------------------------------------
# Copyright (c) 2016 RICOH
# Written by Zou Jinyi
# --------------------------------------------------------
import os.path as osp
import sys
import cv2

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..','..', 'lib')
add_path(lib_path)
import numpy as np
import os
from utils.cython_bbox import bbox_overlaps
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
    return {'boxes' : boxes,
            'gt_classes': gt_classes,
            'flipped' : False,
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

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def scale_and_ratio(min_size,max_size,image_size):
    image_size_min=min(image_size)
    image_size_max=max(image_size)
    im_scale = float(min_size) / float(image_size_min)
    if np.round(im_scale * image_size_max) > max_size:
        im_scale = float(max_size) / float(image_size_max)
    ratio=float(image_size[0])/float(image_size[1])
    return im_scale, ratio

def generate_all_anchors(anchors,feat_stride,num_anchors,conv_width,conv_height,resize_image_size):
    shift_x = np.arange(0, conv_width) * feat_stride
    shift_y = np.arange(0, conv_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
    A = num_anchors
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    allowed_border = 0
    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >=allowed_border) &
        (all_anchors[:, 1] >=allowed_border) &
        (all_anchors[:, 2] < resize_image_size[0] + allowed_border) &  # width
        (all_anchors[:, 3] < resize_image_size[1] + allowed_border)    # height
    )[0]
    anchors = all_anchors[inds_inside, :]
    return anchors
#=======================================================================================#

if __name__ == '__main__':
    data_path='/home/bsl/py-faster-rcnn-master/data/VOCdevkit2007/VOC2007'
    proposal_path='/home/bsl/py-faster-rcnn-master/output/faster_rcnn_alt_opt/voc_2007_trainval/vgg16_rpn_stage2_iter_80000_proposals.pkl'
    image_set='trainval'
    image_set_test='test'
    image_ext = '.jpg'
    image_index=load_image_set_index(data_path,image_set)
    gt_roidb = [load_pascal_annotation(data_path,index) for index in image_index]
    image_test_index = load_image_set_index(data_path,image_set_test)
    test_gt_roidb = [load_pascal_annotation(data_path,index) for index in image_test_index]
    num_images=len(gt_roidb)
    gt_roidb = append_flipped_images(num_images, gt_roidb)

    min_size=600
    max_size=1000
    feat_stride=16
    recall=np.zeros(10,dtype=np.float32)
    im_scale=np.zeros(len(gt_roidb),dtype=np.float32)
    ratio=np.zeros(len(gt_roidb),dtype=np.float32)
    anchors = generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6))
    index=0
    color_gt=(255,0,0)
    color_anchor=(0,255,0)
    # ##==============================================================## anchor recall
    # for j in np.arange(0.1,1.1,0.1):
    #     ind_nums=0
    #     recall_nums=0
    #     dictionary=os.path.join(data_path,'analysis',str(index))
    #     if not os.path.exists(dictionary):
    #         os.mkdir(dictionary)
    #     for i in range(len(gt_roidb)):
    #         if i>=len(gt_roidb)/2:
    #             image_ind=i-len(gt_roidb)/2
    #         else:
    #             image_ind=i
    #         path=os.path.join(data_path,'JPEGImages',image_index[image_ind]+image_ext)
    #
    #         image_size=gt_roidb[i]['size']
    #         im_scale[i],ratio[i] = scale_and_ratio(min_size,max_size,image_size)
    #         resize_image_size = im_scale[i]*np.array(image_size)
    #         conv_width = resize_image_size[0]/feat_stride
    #         conv_height = resize_image_size[1]/feat_stride
    #         projection_gt = np.array(gt_roidb[i]['boxes'])*im_scale[i]
    #         num_anchors=len(anchors)
    #         all_anchors=generate_all_anchors(anchors,feat_stride,num_anchors,conv_width,conv_height,resize_image_size)
    #         overlaps = bbox_overlaps(
    #             np.ascontiguousarray(all_anchors, dtype=np.float),
    #             np.ascontiguousarray(projection_gt, dtype=np.float))
    #         max_overlaps = overlaps.max(axis=0)
    #         argmax_overlaps = overlaps.argmax(axis=0)
    #         im_file = os.path.join(path)
    #         im = cv2.imread(im_file)
    #         if gt_roidb[i]['flipped']:
    #             im = im[:, ::-1, :]
    #             write_path=os.path.join(dictionary,image_index[image_ind]+'_filp'+image_ext)
    #         else:
    #             write_path=os.path.join(dictionary,image_index[image_ind]+image_ext)
    #         img=cv2.resize(im,(int(resize_image_size[0]),int(resize_image_size[1])))
    #         for ii in range(len(max_overlaps)):
    #             boo=False
    #             if max_overlaps[ii]<j:
    #                 boo=True
    #                 rect_start=(int(projection_gt[ii][0]),int(projection_gt[ii][1]))
    #                 rect_end=(int(projection_gt[ii][2]),int(projection_gt[ii][3]))
    #                 cv2.rectangle(img, rect_start, rect_end, color_gt, 2)
    #                 ind=argmax_overlaps[ii]
    #                 anchor_start=(int(all_anchors[ind][0]),int(all_anchors[ind][1]))
    #                 anchor_end=(int(all_anchors[ind][2]),int(all_anchors[ind][3]))
    #                 cv2.rectangle(img, anchor_start, anchor_end, color_anchor, 2)
    #         if boo:
    #             cv2.imwrite(write_path,img)
    #
    #         recall_num=len(np.where(max_overlaps>=j)[0])
    #         ind_nums+=len(projection_gt)
    #         recall_nums+=recall_num
    #     recall[index]=recall_nums/float(ind_nums)
    #     index+=1
    # print 'Anchor_Recall: {}'.format(recall)
    #
    #
    # ##=======================================================## generation proposals recall
    # if os.path.exists(proposal_path):
    #     with open(proposal_path, 'rb') as fid:
    #         proposal_roidb = cPickle.load(fid)
    # index=0
    # for j in np.arange(0.1,1.1,0.1):
    #     ind_nums=0
    #     recall_nums=0
    #     for i in range(len(proposal_roidb)):
    #         image_size=gt_roidb[i]['size']
    #         im_scale[i],ratio[i] = scale_and_ratio(min_size,max_size,image_size)
    #         projection_gt = np.array(gt_roidb[i]['boxes'])*im_scale[i]
    #         proposal = np.array(proposal_roidb[i])*im_scale[i]
    #         overlaps = bbox_overlaps(
    #             np.ascontiguousarray(proposal, dtype=np.float),
    #             np.ascontiguousarray(projection_gt, dtype=np.float))
    #         max_overlaps = overlaps.max(axis=0)
    #         recall_num=len(np.where(max_overlaps>=j)[0])
    #         ind_nums+=len(projection_gt)
    #         recall_nums+=recall_num
    #     recall[index]=recall_nums/float(ind_nums)
    #     index+=1
    # print 'Proposal_Recall: {}'.format(recall)
    # # #=================================================================================## test analysis
    im_scale_test = np.zeros(len(test_gt_roidb),dtype=np.float32)
    test_ratio=np.zeros(len(test_gt_roidb),dtype=np.float32)
    # ##=======================================================## detection proposals recall
    # detection_path = '/home/bsl/py-faster-rcnn-master/output/faster_rcnn_alt_opt/voc_2007_test/VGG16_faster_rcnn_final/proposals.pkl'
    # if os.path.exists(detection_path):
    #     with open(detection_path, 'rb') as fid:
    #         detection_roidb = cPickle.load(fid)
    # index = 0
    # for j in np.arange(0.1, 1.1, 0.1):
    #     ind_nums = 0
    #     recall_nums = 0
    #     dictionary=os.path.join(data_path,'test_analysis',str(index))
    #     if not os.path.exists(dictionary):
    #         os.mkdir(dictionary)
    #     for i in range(len(detection_roidb)):
    #         image_size = test_gt_roidb[i]['size']
    #         im_scale_test[i],test_ratio[i] = scale_and_ratio(min_size,max_size,image_size)
    #         detection_gt = np.array(test_gt_roidb[i]['boxes'])*im_scale_test[i]
    #         detection = np.array(detection_roidb[i])*im_scale_test[i]
    #         overlaps = bbox_overlaps(
    #             np.ascontiguousarray(detection, dtype=np.float),
    #             np.ascontiguousarray(detection_gt, dtype=np.float))
    #         max_overlaps = overlaps.max(axis = 0)
    #         #=========================================================================
    #         argmax_overlaps = overlaps.argmax(axis=0)
    #         test_path=os.path.join(data_path,'JPEGImages',image_test_index[i]+image_ext)
    #         im_file = os.path.join(test_path)
    #         im = cv2.imread(im_file)
    #         write_path=os.path.join(dictionary,image_test_index[i]+image_ext)
    #         resize_test_image_size = im_scale_test[i]*np.array(image_size)
    #         img=cv2.resize(im,(int(resize_test_image_size[0]),int(resize_test_image_size[1])))
    #         #==========================================================================
    #         for ii in range(len(max_overlaps)):
    #             boo=False
    #             if max_overlaps[ii]<j:
    #                 boo=True
    #                 rect_start=(int(detection_gt[ii][0]),int(detection_gt[ii][1]))
    #                 rect_end=(int(detection_gt[ii][2]),int(detection_gt[ii][3]))
    #                 cv2.rectangle(img, rect_start, rect_end, color_gt, 2)
    #                 ind=argmax_overlaps[ii]
    #                 proposal_start=(int(detection[ind][0]),int(detection[ind][1]))
    #                 proposal_end=(int(detection[ind][2]),int(detection[ind][3]))
    #                 cv2.rectangle(img, proposal_start, proposal_end, color_anchor, 2)
    #         if boo:
    #             cv2.imwrite(write_path,img)
    #         #=========================================================================
    #         recall_num = len(np.where(max_overlaps >= j)[0])
    #         ind_nums += len(detection_gt)
    #         recall_nums += recall_num
    #     recall[index] = recall_nums / float(ind_nums)
    #     index+=1
    # print 'Detection_proposal_Recall: {}'.format(recall)

    #============================================================================================predict recall based on class scores and proposals
    predict_path = '/home/bsl/py-faster-rcnn-master/output/faster_rcnn_alt_opt/voc_2007_test/VGG16_faster_rcnn_final/predicts.pkl'
    predict_scores_path = '/home/bsl/py-faster-rcnn-master/output/faster_rcnn_alt_opt/voc_2007_test/VGG16_faster_rcnn_final/predict_scores.pkl'
    if os.path.exists(predict_path):
        with open(predict_path, 'rb') as fid:
            predict_roidbs = cPickle.load(fid)
    if os.path.exists(predict_scores_path):
        with open(predict_scores_path, 'rb') as fid:
            predict_scores = cPickle.load(fid)
    #============================================================================================predict recall based on class scores
    index=0
    for j in np.arange(0.1, 1.1, 0.1):
        dictionary=os.path.join(data_path,'test_predict_analysis',str(index))
        if not os.path.exists(dictionary):
            os.mkdir(dictionary)
        ind_nums = 0
        recall_nums = 0
        for i in range(len(predict_roidbs)):
            test_path=os.path.join(data_path,'JPEGImages',image_test_index[i]+image_ext)
            im_file = os.path.join(test_path)
            im = cv2.imread(im_file)
            write_path=os.path.join(dictionary,image_test_index[i]+image_ext)
            image_size = test_gt_roidb[i]['size']
            im_scale_test[i],test_ratio[i] = scale_and_ratio(min_size,max_size,image_size)
            resize_test_image_size = im_scale_test[i]*np.array(image_size)
            img=cv2.resize(im,(int(resize_test_image_size[0]),int(resize_test_image_size[1])))
            im_scale_test[i],test_ratio[i] = scale_and_ratio(min_size,max_size,image_size)
            detection_gt = np.array(test_gt_roidb[i]['boxes'])*im_scale_test[i]
            argmax_scores = predict_scores[i].argmax(axis=1)
            gt_classes=test_gt_roidb[i]['gt_classes']
            object_ind=np.where(argmax_scores>0)[0]
            predicts=[]
            for ind in range(len(argmax_scores)):
                if argmax_scores[ind]>0:
                    predicts.append(predict_roidbs[i][ind,argmax_scores[ind]*4:(argmax_scores[ind]+1)*4]*im_scale_test[i])
            if len(predicts):
                overlaps = bbox_overlaps(
                    np.ascontiguousarray(predicts, dtype=np.float),
                    np.ascontiguousarray(detection_gt, dtype=np.float))
                max_overlaps = overlaps.max(axis = 0)
                argmax_overlaps = overlaps.argmax(axis=0)
            #=========================================================================
                for num in range(len(gt_classes)):
                    if len(np.where(overlaps[np.where(argmax_scores[object_ind]==gt_classes[num]),num]>j)[1]):
                        recall_nums += 1
                    else :
                        rect_start=(int(detection_gt[num][0]),int(detection_gt[num][1]))
                        rect_end=(int(detection_gt[num][2]),int(detection_gt[num][3]))
                        cv2.rectangle(img, rect_start, rect_end, color_gt, 2)
                        ind_num=argmax_overlaps[num]
                        predict_start=(int(predicts[ind_num][0]),int(predicts[ind_num][1]))
                        predict_end=(int(predicts[ind_num][2]),int(predicts[ind_num][3]))

                        cv2.rectangle(img, predict_start, predict_end, color_anchor, 2)
                        cv2.imwrite(write_path,img)
            ind_nums += len(detection_gt)
        recall[index] = recall_nums / float(ind_nums)
        index+=1
    print 'Predict_Recall: {}'.format(recall)
