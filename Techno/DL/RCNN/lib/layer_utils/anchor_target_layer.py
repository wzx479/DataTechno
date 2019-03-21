# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from lib.utils.cython_bbox import bbox_overlaps

from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform

#�ú�������ÿ��anchor��Ӧ��ground truth(ǰ��/����������ƫ��ֵ)
def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
    """Same as the anchor target layer in original Fast/er RCNN """
    A = num_anchors
    total_anchors = all_anchors.shape[0]
    K = total_anchors / num_anchors
    im_info = im_info[0]

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]

    # only keep anchors inside the image
    # 2 inds_inside = ���е�anchor��x1,y1,x2,y2û�г���ͼ��߽�ġ�
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    # labels �ֶεĳ��Ⱦ��ǺϷ���anchor�ĸ���
    # ����-1���labels
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes ��
    # 3 ����Ԥ����ֵ��overlap�ص��ʣ�����ǰ������ǩ1|0
    # bbox_overlaps (������anchors��gt_boxes֮����غ϶�IOU������0.7���Ϊǰ��ͼ��С��0.3���Ϊ����ͼ;
    # ��������(n,k),����n��anchors���K��gt_boxes��IOU�غ϶�ֵ
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))

    # 4 ����ÿ��anchor���ҵ���gt_box�����IOU�����ֵ�����ҵ�ÿ��anchors����ص��ʵ�gt_boxes��
    argmax_overlaps = overlaps.argmax(axis=1) # ? #
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps] # ? # anchors��gt_boxes���IoU

    gt_argmax_overlaps = overlaps.argmax(axis=0) # ? #
    gt_max_overlaps = overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])] # ? #

    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]#�ٴζ���ÿ��gt_box���ҵ���Ӧ�����overlap��anchor��shape[len(gt_boxes),]

    if not cfg.FLAGS.rpn_clobber_positives:
        # assign bg labels first so that positive labels can clobber them
        # first set the negatives , �����ｫanchors��gt_boxes���IoU��ȻС����ֵ(0.3)��ĳЩanchor��0
        labels[max_overlaps < cfg.FLAGS.rpn_negative_overlap] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU , �����ｫanchors��gt_boxes���IoU������ֵ(0.7)��ĳЩanchor��1
    labels[max_overlaps >= cfg.FLAGS.rpn_positive_overlap] = 1

    if cfg.FLAGS.rpn_clobber_positives:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.FLAGS.rpn_negative_overlap] = 0

    # 5 �������һЩǰ��anchor�ͱ���anchors. ���β���
    # subsample positive labels if we have too many
    num_fg = int(cfg.FLAGS.rpn_fg_fraction * cfg.FLAGS.rpn_batchsize)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1 #�����ʵ���ڵ�ǰ��anchor����������ֵ�����������һЩǰ��anchor

    # subsample negative labels if we have too many
    num_bg = cfg.FLAGS.rpn_batchsize - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    # 6 ʹ��bbox_transform����������ÿ��anchor������overlap��gt_boxes�Ŀ�ƫ������
    # ��Ϊλ�����ı�ǩֵ(tx,ty,th,tw)���ں�����ع�
    # bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)  # ��ÿ����ԭͼ�ڲ���anchor,��ȫ0��ʼ������任ֵ
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])


    # bbox_inside_weights��bbox_outside_weights��������������ѵ��anchor�߿�����ʱ���ش�����

    #7 �ڽ��б߿�����loss�ļ���ʱ��ֻ��ǰ��anchor�������ã����Կ�������bbox_inside_weights��bbox_outside_weights��ʵ�֡�
    # ��ǰ���ͱ���anchor��Ӧ��bbox_inside_weights��bbox_outside_weights��Ϊ0  @ https://blog.csdn.net/u012426298/article/details/81517609
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # only the positive ones have regression targets
    bbox_inside_weights[labels == 1, :] = np.array(cfg.FLAGS2["bbox_inside_weights"])
    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)


    if cfg.FLAGS.rpn_positive_weight < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((cfg.FLAGS.rpn_positive_weight > 0) &
                (cfg.FLAGS.rpn_positive_weight < 1))
        positive_weights = (cfg.FLAGS.rpn_positive_weight /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.FLAGS.rpn_positive_weight) /
                            np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    # 8.ͳһ���еı�ǩ����ת����ǩlabels�ĸ�ʽ�󣬷���
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0) # �� #
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)  # �� #

    # labels
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_outside_weights = bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

#  _unmap���� ��ͼ���ڲ���anchor ӳ��ص����ɵ����е�anchor
def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

#_compute_targets��������anchor�Ͷ�Ӧ��gt_box��λ��ӳ��
def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
