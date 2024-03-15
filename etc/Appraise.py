import torch
from torch import nn
from work.utils import reverse_one_hot
import torch.nn.functional as F
import numpy as np


# count loss
def loss(logits,labels):
    # return nn.BCEWithLogitsLoss()(logits, lab)

    if logits.shape == labels.shape:
        labels = torch.argmax(labels,dim=1)
    elif len(labels.shape) == 3:
        labels = labels
    else:
        assert "pred.shape not match label.shape"
    #logits = F.softmax(logits,dim=1)
    return nn.CrossEntropyLoss()(logits,labels)

def dice_loss(logits,targets,smooth=1.):
    # logic.shape=[C,N,H,W],target.shape=[C,N,H,W]&[C,1,H,W]
    if logits.shape == targets.shape:
        targets = targets.type(torch.float32)
    elif targets.shape[1] == 1:
        targets = targets.type(torch.int64)
        targets = torch.zeros_like(logits).scatter_(dim=1, index=targets, src=torch.tensor(1.0)).type(torch.float32)

    else:
        assert "pred.shape not match label.shape"

    outputs = F.softmax(logits,dim=1).type(torch.float32)
    #outputs = logits
    #targets=torch.zeros_like(logits).scatter_(dim=1,index=targets,src=torch(1.0))

    inter = outputs*targets
    dice = 1 - ((2 * inter.sum(dim=(2, 3)) + smooth) / (outputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth))
    return dice.mean().item()



def count_mean_out_nan(x,transpose=False):
    assert len(x.shape)==2
    if transpose:
        x = torch.transpose(x,1,0)
    row_list = []
    for clu in x:
        clu_list = []
        for row in clu:
            if not torch.isnan(row):
                clu_list.append(row)
        row_list.append(np.mean(clu_list) if len(clu_list)!=0 else 0)
    if transpose:
        return row_list
    return np.mean(row_list)

# count mean_iou and mean_pa
def mean_iou_pa(logits, targets):
    #logic.shape=[C,N,H,W],target.shape=[C,N,H,W]&[C,1,H,W]
    if logits.shape == targets.shape:
        targets = targets.type(torch.int8)
    elif targets.shape[1] == 1:
        #print("target.shape equal [C,1,H,W]")
        #targets = targets.type(torch.int64)
        targets = torch.zeros_like(logits).scatter_(dim=1, index=targets, src=torch.tensor(1.0)).type(torch.int8)
    elif len(targets.shape)==3:
        targets = torch.unsqueeze(targets,dim=1)
        targets = torch.zeros_like(logits).scatter_(dim=1, index=targets, src=torch.tensor(1.0)).type(torch.int8)
    else:
        assert "pred.shape not match label.shape"

    outputs = torch.argmax(logits, dim=1, keepdim=True).type(torch.int64)
    outputs = torch.zeros_like(logits).scatter_(dim=1, index=outputs, src=torch.tensor(1.0)).type(torch.int8)

    inter = (outputs & targets).type(torch.float32).sum(dim=(2, 3))
    union = (outputs | targets).type(torch.float32).sum(dim=(2, 3))
    iou = inter / union

    return count_mean_out_nan(iou.cpu())#iou.mean().item() #, pa.mean().item()

def iou_mpa(logits, targets):
    #logic.shape=[C,N,H,W],target.shape=[C,N,H,W]&[C,1,H,W]
    # print("miou logits",logits.shape)
    # print("miou targets",targets.shape)
    if logits.shape == targets.shape:
        #print("shape equal")
        targets = targets.type(torch.int8)
    elif targets.shape[1] == 1:
        #print("target.shape equal [C,1,H,W]")
        targets = targets.type(torch.int64)
        targets = torch.zeros_like(logits).scatter_(dim=1, index=targets, src=torch.tensor(1.0)).type(torch.int8)
    else:
        assert "pred.shape not match label.shape"

    outputs = torch.argmax(logits, dim=1, keepdim=True).type(torch.int64)
    #targets = torch.argmax(targets,dim=1,keepdim=True).type(torch.int64)
    #targets = targets.type(torch.int8)
    outputs = torch.zeros_like(logits).scatter_(dim=1, index=outputs, src=torch.tensor(1.0)).type(torch.int8)
    #targets = torch.zeros_like(logits).scatter_(dim=1, index=targets, src=torch.tensor(1.0)).type(torch.int8)

    inter = (outputs & targets).type(torch.float32).sum(dim=(2, 3))
    union = (outputs | targets).type(torch.float32).sum(dim=(2, 3))
    iou = inter / union

    # iou = torch.where(torch.isnan(iou), torch.full_like(iou, 1), iou)
    # iou = iou.mean(dim=0).squeeze().cpu()

    targets_float = targets.type(torch.float32).sum(dim=(2, 3))
    pa = inter/targets_float
    mpa = count_mean_out_nan(pa.cpu())

    return iou, mpa


if __name__ == "__main__":
    print("Appraise run")
    from dataset.utils import imshow
    from dataset.Reader import Reader
    from work.utils import colour_code_segmentation
    from PIL import Image
    import cv2
    from work.utils import one_hot_it

    label_class = {'void': [0, 0, 0], 'build': [38, 38, 38]}
    dara_dir = "../3labels/test/1.png"
    image = cv2.imread(dara_dir)
    image = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2RGB)
    #img = np.argmax(image,axis=-1)
    lab_tmp = one_hot_it(image,label_class).astype(np.uint8)
    #lab_tmp = np.argmax(lab_tmp,axis=-1)
    res = np.sum(lab_tmp,axis=(0,1))
    print(res[0]/res[1])







