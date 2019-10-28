from PIL import Image
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List, Tuple
torch.backends.cudnn.benchmark = True

import os,sys,cv2,random,datetime,time,math
import argparse
import numpy as np

from timebudget import timebudget

from .bbox import decode, nms
from .net_s3fd import S3fd_Model

from autojit import autojit

# It's a trade-off.  Not really clear which is faster or why.
# On CPU, all the time (~40ms) is spent moving the data from GPU to CPU. 
# On GPU, all the time (~40ms) is spent on the *first* torch.nonzero for some reason.
use_cpu_for_decoding_bbox = True

def detect_faces(net:nn.Module, img:np.ndarray, minscale:int=3, ovr_threshhold:float=0.3,
                 score_threshhold:float=0.5) -> List[Tuple]:
    """returns an list of tuples describing bounding boxes: [x1,y1,x2,y2,score].
    Setting minscale to 0 finds the smallest faces, but takes the longest.
    """
    bboxlist = detect(net, img, minscale)
    keep_idx = nms(bboxlist, ovr_threshhold)
    bboxlist = bboxlist[keep_idx,:]
    out = []
    for b in bboxlist:
        x1,y1,x2,y2,s = b
        if s<0.5: 
            continue
        out.append((int(x1),int(y1),int(x2),int(y2),s))
    return out

@timebudget
def _process_bbox(i:torch.Tensor, ocls:torch.Tensor, oreg:torch.Tensor, hw:torch.Tensor) -> Tuple[torch.Tensor]:
    stride = 2**(i+2)    # 4,8,16,32,64,128
    anchor = stride*4
    hindex = hw[0]
    windex = hw[1]
    score = ocls[0,1,hindex,windex]
    loc = oreg[0,:,hindex,windex].contiguous().view(1,4)
    variances = torch.Tensor([0.1,0.2])
    return _process_bbox2(stride, anchor, score, loc, hindex, windex, variances)

@autojit  # this gives a warning.  is it okay?
def _process_bbox2(stride, anchor, score, loc, hindex, windex, variances):
    axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
    priors = torch.cat([axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]).unsqueeze(0)
    if not use_cpu_for_decoding_bbox:
        priors = priors.cuda()
        variances = variances.cuda()
    box = decode(loc,priors,variances)
    x1,y1,x2,y2 = box[0]*1.0
    return (x1,y1,x2,y2,score)

@timebudget
def bboxlist_from_olist(olist:List[torch.Tensor], minscale:int=3) -> torch.Tensor:
    bboxlist = []
    for i in range(minscale, len(olist)//2):
        ocls = F.softmax(olist[i*2], dim=1).data
        oreg = olist[i*2+1].data
        FB,FC,FH,FW = ocls.size() # feature map size
        stride = 2**(i+2)    # 4,8,16,32,64,128
        anchor = stride*4
        all_scores = ocls[0,1,:,:]
        if use_cpu_for_decoding_bbox:
            with timebudget('move-to-cpu'):
                all_scores = all_scores.cpu()
                oreg = oreg.cpu()
        # instead of running a sliding window, first find the places where score is big enough to bother
        with timebudget('scan-bigenough'): 
            # For some reason, this is crazy slow (38ms) on GPU, but only the first time it's called.
            bigenough = torch.nonzero(all_scores > 0.05)
        for hw in bigenough:
            i_t = torch.ones(1) * i
            bboxlist.append(_process_bbox(i_t, ocls, oreg, hw))
    if len(bboxlist) == 0: 
        bboxlist=torch.zeros((1, 5))
    bboxlist = torch.Tensor(bboxlist)
    return bboxlist


def olist_from_img(net:nn.Module, img:np.ndarray) -> List[torch.Tensor]:
    img = img - np.array([104,117,123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,)+img.shape)

    img = Variable(torch.from_numpy(img).float()).cuda()
    olist = net(img)
    return olist

def detect(net:nn.Module, img:np.ndarray, minscale:int=3) -> torch.Tensor:
    """returns an Nx5 tensor describing bounding boxes: [x1,y1,x2,y2,score].
    This will have LOTS of similar/overlapping regions.  Need to call bbox.nms to reconcile them.
    Setting minscale to 0 finds the smallest faces, but takes the longest.
    """
    olist = olist_from_img(net, img)
    return bboxlist_from_olist(olist, minscale)
    

