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

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        
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
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
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
        
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        #wx na代表每个特征图的anchors数量，在这里为3
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
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
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
