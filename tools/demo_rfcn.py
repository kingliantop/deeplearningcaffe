#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import json

CLASSES = ('__background__',
           'hsb-tianshui-mianrou360', 'hsb-tianshui-mianrou400', 'hsb-tianshui-ganshuang360',
           'hsb-tianshui-ganshuang400', 'sj', 'ygm', 'sufei-chaoshushui-mianrou290',
           'sf-chaoshushui-mianrou350', 'sf-chaoshushui-ganshuang290', 'sf-chaoshushui-ganshuang350',
           'sf-chaoshushui-420', 'sf-litihuwei', 'sf-tanlitieshen', 'sf-koudaimofa', 'gjs-riyong(hong)',
           'gjs-yeyong(lan)', 'qdkj-gongzhuxilie', 'qdkj-shaonvxilie', 'qdkj-youyaxilie',
           'abc-riyong(lv)', 'abc-yeyong(zi)', 'lb-kispa hbb-<1kg', 'lb-kispa hbb->=1kg',
           'lb-putong-<1kg', 'lb-putong->=1kg', 'tz<1kg', 'tz>=1kg', 'am-quanzidong-<1kg',
           'am-quanzidong->=1kg', 'am-jinglanquanxiao-<1kg', 'am-jinglanquanxiao->=1kg', 'pl<1kg',
           'pl>=1kg', 'xyy-lb', 'xxy-tz', 'xxy-bl', 'pt', 'pr', 'sy', 'shk')
#           'kkkl-kkkele-<=600ml', 'kkkl-kkkele->600ml', 'kkkl-kkkele-can',
#           'kkkl-xuebi-<=600ml', 'kkkl-xuebi->600ml', 'kkkl-xuebi-can',
#           'kkkl-fenda-<=600ml', 'kkkl-fenda->600ml', 'kkkl-fenda-can',
#           'kkkl-yiquan+c-<=600ml', 'kkkl-yiquan+c-can',
#           'kkkl-meizhiyuan(ny)-<=600ml',
#           'kkkl-meizhiyuan(ny)->600ml', 'kkkl-meizhiyuan(gz)-<=600ml',
#           'kkkl-meizhiyuan(gz)->600ml', 'ps-pskele-<=600ml',
#           'ps-pskele->600ml',
#           'ps-pskele-can', 'ps-meinianda-<=600ml', 'ps-meinianda->600ml',
#           'ps-meinianda-can', 'ksf-lvcha-<=600ml', 'ksf-binghongcha-<=600ml',
#           'ksf-binglvcha-<=600ml', 'dn-maidong-<=600ml', 'kkkl-binglu-sspet',
#           'kkkl-chunyue-sspet', 'nfsq-nongfushanquan-sspet')
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'ResNet-101': ('ResNet-101',
                  'resnet101_rfcn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
		  'resnet50_rfcn_ohem_iter_100000.caffemodel')}
#                  'resnet50_rfcn_final.caffemodel')}

#define a function to generate detect result from the model
#added by stevenL 2017-3-13
def generateResultImage(im, image_name, paraList, thresh=0.5):

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    results = []
    for para in paraList:
        for class_name, dets in para.items():
            inds = np.where(dets[:, -1] >= thresh)[0]
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                print('class name:' + class_name + ' score:' + str(score))
                #print('The Box:' + str(bbox[0]) + ', ' + str(bbox[1]) + ', '
                #      + str(bbox[2]) + ', ' + str(bbox[3]))
                results.append({'labelname':class_name,
                                'possibility':str(score),
                                'values':{'xmin':str(bbox[0]),
                                          'ymin':str(bbox[1]),
                                          'xmax':str(bbox[2]),
                                          'ymax':str(bbox[3])}})
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor='red', linewidth=3.5)
                )
                ax.text(bbox[0], bbox[1] - 2,
                        '{:s} {:.3f}'.format(class_name, score),
                        bbox=dict(facecolor='blue', alpha=0.5),
                        fontsize=14, color='white')

    ax.set_title(('{} detections with '
                        'p({} | box) >= {:.1f}').format(image_name, image_name,
                                                  thresh),fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #save the result to a file
    tag_name = image_name[:-4]+'-tag.jpg'
    json_filename = image_name[:-4]+'.json'

    tag_file = os.path.join(cfg.DATA_DIR, 'tags', tag_name)
    json_file = os.path.join(cfg.DATA_DIR, 'json', json_filename)
    #plt.show()
    plt.savefig(tag_file)
    with open(json_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)
    outfile.close()

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        print('class name:'+class_name+' score:' + str(score))
        print('The Box:' + str(bbox[0]) + ', ' + str(bbox[1]) + ', '
                + str(bbox[2]) + ', '+ str(bbox[3]))
	    #print(score)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print("working om image:" + image_name)
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    paraList = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #steven lian: pre-check the valid Boxes and classes
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) != 0:
            paraList.append({cls:dets})
        #Just save the class name and dets to an dict in a list
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)
    generateResultImage(im, image_name, paraList, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        choices=NETS.keys(), default='ResNet-101')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'rfcn_end2end', 'test_agnostic.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'rfcn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    #im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    #for i in xrange(2):
    #    _, _= im_detect(net, im)

    demopath = os.path.join(cfg.DATA_DIR, 'demo')
    im_names = sorted([x for x in os.listdir(demopath) if not x.startswith('.') and x.endswith('.jpg')])

    #print(filenames)
#    amount = len(filenames)
    #im_names = ['004494.jpg', '2008_005681.jpg', '2009_003955.jpg',
	#        	'2012_002437.jpg', '2012_004306.jpg']

    #im_names = ['4_P00001_299_0502598163_1171_20161209135333.jpg',
    #            '4_P00001_299_0502601195_1171_20161210110609.jpg',
    #            '4_P00001_299_0502207022_1172_20161209100357.jpg']

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)
