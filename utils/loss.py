# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel

import ipdb

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets, imgs):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        
        #深度眸 可视化target
        vis_bbox(imgs, targets)

        #wx type(p)=<class 'list'>, len(p)=3
        #wx type(p[0])=<class 'torch.Tensor'>, p[0].shape=torch.Size([BS, 3, 80, 80, 85]), 640 / 8
        #wx type(p[1])=<class 'torch.Tensor'>, p[1].shape=torch.Size([BS, 3, 40, 40, 85]), 640 / 16
        #wx type(p[2])=<class 'torch.Tensor'>, p[2].shape=torch.Size([BS, 3, 20, 20, 85]), 640 / 32
        #wx type(targets)=<class 'torch.Tensor'>, targets.shape=torch.Size([objNum, 6])
        #wx targets.shape[0]代表这个batch图片集中的objects数量
        #wx targets.shape[1]代表batchNo, cls, xc_relative, yc_relative, w_relative, h_relative
        #wx type(tcls)=<class 'list'>, len(tcls)=3,
        #wx type(tcls[0])=<class 'torch.Tensor'>, tcls[0].shape=torch.Size([45])
        #wx type(tcls[1])=<class 'torch.Tensor'>, tcls[1].shape=torch.Size([102])
        #wx type(tcls[2])=<class 'torch.Tensor'>, tcls[2].shape=torch.Size([84])
        #wx type(tbox)=<class 'list'>, len(tbox)=3
        #wx type(tbox[0])=<class 'torch.Tensor'>, tbox[0].shape=torch.Size([45, 4])
        #wx type(tbox[1])=<class 'torch.Tensor'>, tbox[1].shape=torch.Size([102, 4])
        #wx type(tbox[2])=<class 'torch.Tensor'>, tbox[2].shape=torch.Size([84, 4])
        #wx type(indices)=<class 'list'>, len(indices)=3
        #wx type(indices[0])=<class 'tuple'>, len(indices[0])=4
        #wx type(indices[0][0])=<class 'torch.Tensor'>, indices[0][0].shape=torch.Size([45])
        #wx type(indices[0][1])=<class 'torch.Tensor'>, indices[0][1].shape=torch.Size([45])
        #wx type(indices[0][2])=<class 'torch.Tensor'>, indices[0][2].shape=torch.Size([45])
        #wx type(indices[0][3])=<class 'torch.Tensor'>, indices[0][3].shape=torch.Size([45])
        #wx type(indices[1])=<class 'tuple'>, len(indices[1])=4
        #wx type(indices[1][0])=<class 'torch.Tensor'>, indices[1][0].shape=torch.Size([102])
        #wx type(indices[1][1])=<class 'torch.Tensor'>, indices[1][1].shape=torch.Size([102])
        #wx type(indices[1][2])=<class 'torch.Tensor'>, indices[1][2].shape=torch.Size([102])
        #wx type(indices[1][3])=<class 'torch.Tensor'>, indices[1][3].shape=torch.Size([102])
        #wx type(indices[2])=<class 'tuple'>, len(indices[2])=4
        #wx type(indices[2][0])=<class 'torch.Tensor'>, indices[2][0].shape=torch.Size([84])
        #wx type(indices[2][1])=<class 'torch.Tensor'>, indices[2][1].shape=torch.Size([84])
        #wx type(indices[2][2])=<class 'torch.Tensor'>, indices[2][2].shape=torch.Size([84])
        #wx type(indices[2][3])=<class 'torch.Tensor'>, indices[2][3].shape=torch.Size([84])
        #wx type(anchors)=<class 'list'>, len(anchors)=3
        #wx type(anchors[0])=<class 'torch.Tensor'>, anchors[0].shape=torch.Size([45, 2])
        #wx type(anchors[1])=<class 'torch.Tensor'>, anchors[1].shape=torch.Size([102, 2])
        #wx type(anchors[2])=<class 'torch.Tensor'>, anchors[2].shape=torch.Size([84, 2])
        tcls, tbox, indices, anchors, ttars = self.build_targets(p, targets)  # targets

        #深度眸 可视化anchor匹配关系
        vis_match(imgs, targets, tcls, tbox, indices, anchors, p, ttars)

        ipdb.set_trace()

        # Losses
        #wx p是特征图, P3/P4/P5, len(p)=3
        #wx p[0].shape=torch.Size([2, 3, 80, 80, 85]), 浅层大特征图, 有利于检测小目标
        #wx p[1].shape=torch.Size([2, 3, 40, 40, 85])
        #wx p[2].shape=torch.Size([2, 3, 20, 20, 85]), 浅层小特征图, 有利于检测大目标
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    #深度眸 用于计算loss函数所需要的target
    #wx 正负样本分配
    def build_targets(self, p, targets):
        #wx p是特征图, P3/P4/P5, len(p)=3
        #wx type(p)=<class 'list'>, len(p)=3
        #wx type(p[0])=<class 'torch.Tensor'>, p[0].shape=torch.Size([BS, 3, 80, 80, 85]), 640 / 8
        #wx type(p[1])=<class 'torch.Tensor'>, p[1].shape=torch.Size([BS, 3, 40, 40, 85]), 640 / 16
        #wx type(p[2])=<class 'torch.Tensor'>, p[2].shape=torch.Size([BS, 3, 20, 20, 85]), 640 / 32
        #wx type(targets)=<class 'torch.Tensor'>, targets.shape=torch.Size([objs_num, 6])
        #wx targets.shape[0]代表这个batch训练集中的objects数量objs_num
        #wx targets.shape[1]代表batchNo, cls, xc_relative, yc_relative, w_relative, h_relative
        #wx batchNo表示这条label信息属于该batch训练集中的第几张图片

        
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        #wx na代表每个特征图的anchors数量，在这里为3
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, ttars = [], [], [], [], []
        #wx type(gain)=<class 'torch.Tensor'>, gain.shape=torch.Size([7])
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        #wx type(ai)=<class 'torch.Tensor'>, ai.shape=torch.Size([3, 14])
        #wx tensor([[0.], [1.], [2.]], device='cuda:0')
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        #wx targets.repeat(na, 1, 1).shape=torch.Size([3, 14, 6])
        #wx targets(image, class, xc_relative, yc_relative, w_relative, h_relative, anchor indices)
        #wx targets.shape=torch.Size([3, 14, 7])
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        #wx off.shape=torch.Size([5, 2])
        #wx 分别对应中心点、左、上、右、下
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            #wx self.anchors.shape=torch.Size([3, 3, 2]), (P3/P4/P5, num of anchors, w/h)
            #wx p[0].shape=torch.Size([BS, 3, 80, 80, 85])
            #wx p[1].shape=torch.Size([BS, 3, 40, 40, 85])
            #wx p[2].shape=torch.Size([BS, 3, 20, 20, 85])
            anchors, shape = self.anchors[i], p[i].shape
            #wx gain=tensor([ 1.,  1., 80., 80., 80., 80.,  1.], device='cuda:0')
            #wx gain = [1, 1, 特征图w, 特征图_h, 特征图w, 特征图_h, 1]
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            #wx t.shape=torch.Size([3, objs_num, 7]), targets代表gt
            t = targets * gain  # shape(3,n,7)
            #wx nt代表这个batchs训练数据的目标个数
            if nt:
                ipdb.set_trace()
                # Matches
                #wx anchors.shape=torch.Size([3, 2])，anchors[:, None].shape=torch.Size([3, 1, 2])
                #wx t[..., 4:6].shape=torch.Size([3, objs_num, 2])
                #wx r.shape=torch.Size([3, objs_num, 2])
                #wx t[:, :, 4:6]取到的是gt的宽高
                #wx 所有的gt的宽高与anchors的宽高计算比例 
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                #wx 默认self.hyp['anchor_t']=4，代表target宽高与anchor宽高比例都必须处于1/4到4区间内，才能与当前anchor匹配
                #wx j.shape=torch.Size([3, 14])
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                #wx t代表保留的gt, 到这里后只保留了能与当前特征图的3个anchors匹配的gt, 每层特征图是3个anchor
                #wx t.shape=torch.Size([34, 7])
                t = t[j]  # filter

                # Offsets
                #wx t各个维度的含义是(image, class, xc_relative, yc_relative, w_relative, h_relative, anchor indices)
                gxy = t[:, 2:4]  # grid xy
                #wx gain = [1, 1, 特征图w, 特征图_h, 特征图w, 特征图_h, 1]
                gxi = gain[[2, 3]] - gxy  # inverse
                #wx jklm就分别代表左、上、右、下是否能作为正样本。g=0.5
                #wx j和l, k和m是互斥的, j.shape=torch.Size([34])
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                #wx j.shape=torch.Size([5, 34])
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                #wx 原本一个gt只会存储一份，现在复制成5份
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            ttars.append(torch.cat((gxy, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch, ttars


##################################################################################
#深度眸 可视化bbox
import numpy as np
import cv2

def vis_bbox(imgs, targets):
    tar = targets.cpu().detach().numpy()
    data = imgs * 255
    data = data.permute(0, 2, 3, 1).cpu().detach().numpy()
    h, w = data.shape[1], data.shape[1]
    gain = np.ones(6)
    gain[2:6] = np.array([w, h, w, h])
    tar = (tar * gain)
    for i in range(imgs.shape[0]):
        img = data[i].astype(np.uint8)
        img = img[..., ::-1]
        tar1 = tar[tar[:, 0] == i][:, 2:]
        y = xywhToxyxy(tar1)
        show_bbox(img, y)

def vis_match(imgs, targets, tcls, tboxs, indices, anchors, pred, ttars):
    tar = targets.cpu().detach().numpy()
    data = imgs * 255
    data = data.permute(0, 2, 3, 1).cpu().detach().numpy()
    h, w = data.shape[1], data.shape[2]
    gain = np.ones(6)
    gain[2:6] = np.array([w, h, w, h])
    tar = (tar * gain)

    strdie = [8, 16, 32]
    # 对每张图片进行可视化
    for j in range(imgs.shape[0]):
        img = data[j].astype(np.uint8)[..., ::-1]
        tar1 = tar[tar[:, 0] == j][:, 2:]
        y1 = xywhToxyxy(tar1)
        # img = VisualHelper.show_bbox(img1.copy(), y1, color=(255, 255, 255), is_show=False, thickness=2)
        # 对每个预测尺度进行单独可视化
        vis_imgs = []
        for i in range(3):  # i=0检测小物体，i=1检测中等尺度物体，i=2检测大物体
            s = strdie[i]
            # anchor尺度
            gain1 = np.array(pred[i].shape)[[3, 2, 3, 2]]
            b, a, gx, gy = indices[i]
            b1 = b.cpu().detach().numpy()
            gx1 = gx.cpu().detach().numpy()
            gy1 = gy.cpu().detach().numpy()
            anchor = anchors[i].cpu().detach().numpy()
            ttar = ttars[i].cpu().detach().numpy()

            # 找出对应图片对应分支的信息
            indx = b1 == j
            gx1 = gx1[indx]
            gy1 = gy1[indx]
            anchor = anchor[indx]
            ttar = ttar[indx]

            # 还原到原图尺度进行可视化
            ttar /= gain1
            ttar *= np.array([w, h, w, h], np.float32)
            y = xywhToxyxy(ttar)
            # label 可视化
            img1 = show_bbox(img.copy(), y, color=(0, 0, 255), is_show=False)

            # anchor 需要考虑偏移，在任何一层，每个bbox最多3*3=9个anchor进行匹配
            anchor *= s
            anchor_bbox = np.stack([gy1, gx1], axis=1)
            k = np.array(pred[i].shape, np.float)[[3, 2]]
            anchor_bbox = anchor_bbox / k
            anchor_bbox *= np.array([w, h], np.float32)
            anchor_bbox = np.concatenate([anchor_bbox, anchor], axis=1)
            anchor_bbox1 = xywhToxyxy(anchor_bbox)
            # 正样本anchor可视化
            img1 = show_bbox(img1, anchor_bbox1, color=(0, 255, 255), is_show=False)
            vis_imgs.append(img1)
        show_img(vis_imgs, is_merge=True)

def xywhToxyxy(bbox):
    y = np.zeros_like(bbox)
    y[:, 0] = bbox[:, 0] - bbox[:, 2] / 2  # top left x
    y[:, 1] = bbox[:, 1] - bbox[:, 3] / 2  # top left y
    y[:, 2] = bbox[:, 0] + bbox[:, 2] / 2  # bottom right x
    y[:, 3] = bbox[:, 1] + bbox[:, 3] / 2  # bottom right y
    return y

def show_bbox(image, bboxs_list, color=None,
              thickness=1, font_scale=0.3, wait_time_ms=0, names=None,
              is_show=True, is_without_mask=False):
    """
    Visualize bbox in object detection by drawing rectangle.

    :param image: numpy.ndarray.
    :param bboxs_list: list: [pts_xyxy, prob, id]: label or prediction.
    :param color: tuple.
    :param thickness: int.
    :param fontScale: float.
    :param wait_time_ms: int
    :param names: string: window name
    :param is_show: bool: whether to display during middle process
    :return: numpy.ndarray
    """
    assert image is not None
    font = cv2.FONT_HERSHEY_SIMPLEX
    image_copy = image.copy()
    for bbox in bboxs_list:
        if len(bbox) == 5:
            txt = '{:.3f}'.format(bbox[4])
        elif len(bbox) == 6:
            txt = 'p={:.3f},id={:.3f}'.format(bbox[4], bbox[5])
        bbox_f = np.array(bbox[:4], np.int32)
        if color is None:
            colors = random_color(rgb=True).astype(np.float64)
        else:
            colors = color

        if not is_without_mask:
            image_copy = cv2.rectangle(image_copy, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors,
                                       thickness)
        else:
            mask = np.zeros_like(image_copy, np.uint8)
            mask1 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, -1)
            mask = np.zeros_like(image_copy, np.uint8)
            mask2 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, thickness)
            mask2 = cv2.addWeighted(mask1, 0.5, mask2, 8, 0.0)
            image_copy = cv2.addWeighted(image_copy, 1.0, mask2, 0.6, 0.0)
        if len(bbox) == 5 or len(bbox) == 6:
            cv2.putText(image_copy, txt, (bbox_f[0], bbox_f[1] - 2),
                        font, font_scale, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
    if is_show:
        show_img(image_copy, names, wait_time_ms)
    return image_copy

def show_img(imgs, window_names=None, wait_time_ms=0, is_merge=False, row_col_num=(1, -1)):
    """
        Displays an image or a list of images in specified windows or self-initiated windows.
        You can also control display wait time by parameter 'wait_time_ms'.
        Additionally, this function provides an optional parameter 'is_merge' to
        decide whether to display all imgs in a particular window 'merge'.
        Besides, parameter 'row_col_num' supports user specified merge format.
        Notice, specified format must be greater than or equal to imgs number.

        :param imgs: numpy.ndarray or list.
        :param window_names: specified or None, if None, function will create different windows as '1', '2'.
        :param wait_time_ms: display wait time.
        :param is_merge: whether to merge all images.
        :param row_col_num: merge format. default is (1, -1), image will line up to show.
                            example=(2, 5), images will display in two rows and five columns.
        """
    if not isinstance(imgs, list):
        imgs = [imgs]

    if window_names is None:
        window_names = list(range(len(imgs)))
    else:
        if not isinstance(window_names, list):
            window_names = [window_names]
        assert len(imgs) == len(window_names), 'window names does not match images!'

    if is_merge:
        merge_imgs1 = merge_imgs(imgs, row_col_num)

        cv2.namedWindow('merge', 0)
        cv2.imshow('merge', merge_imgs1)
    else:
        for img, win_name in zip(imgs, window_names):
            if img is None:
                continue
            win_name = str(win_name)
            cv2.namedWindow(win_name, 0)
            cv2.imshow(win_name, img)

    cv2.waitKey(wait_time_ms)

def merge_imgs(imgs, row_col_num):
    """
        Merges all input images as an image with specified merge format.

        :param imgs : img list
        :param row_col_num : number of rows and columns displayed
        :return img : merges img
        """

    length = len(imgs)
    row, col = row_col_num

    assert row > 0 or col > 0, 'row and col cannot be negative at same time!'
    color = random_color(rgb=True).astype(np.float64)

    for img in imgs:
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color)

    if row_col_num[1] < 0 or length < row:
        merge_imgs = np.hstack(imgs)
    elif row_col_num[0] < 0 or length < col:
        merge_imgs = np.vstack(imgs)
    else:
        assert row * col >= length, 'Imgs overboundary, not enough windows to display all imgs!'

        fill_img_list = [np.zeros(imgs[0].shape, dtype=np.uint8)] * (row * col - length)
        imgs.extend(fill_img_list)
        merge_imgs_col = []
        for i in range(row):
            start = col * i
            end = col * (i + 1)
            merge_col = np.hstack(imgs[start: end])
            merge_imgs_col.append(merge_col)

        merge_imgs = np.vstack(merge_imgs_col)

    return merge_imgs

def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000
    ]
).astype(np.float32).reshape(-1, 3)