import torch
import torch.nn as nn
from .backbone import define_F, CDSA

class STANetSA(nn.Module):
    """
    change detection module:
    feature extractor+ spatial-temporal-self-attention
    contrastive loss
    """
    def __init__(self,in_channels, f_c=64, ds=1, SA_mode="PAM"):
        super().__init__()
        self.net_f = define_F(in_channels, f_c)
        self.net_a = CDSA(f_c, ds ,SA_mode)
        self.out_cbn1 = nn.Conv2d(f_c,1,1,1)
        self.out_cbn2 = nn.Conv2d(f_c,1,1,1)
    
    def forward(self, x1, x2=None):
        if x2 == None:
            x1, x2 = x1[:,3,:,:], x1[:,3:,:,:]
        else:
            x1 = x1
            x2 = x2

        feat_A = self.net_f(x1)  # f(A)
        feat_B = self.net_f(x2)   # f(B)

        feat_A, feat_B = self.net_a(feat_A,feat_B)
        feat_A = self.out_cbn1(feat_A)
        feat_B = self.out_cbn2(feat_B)
        
        dist = nn.functional.pairwise_distance(feat_A, feat_B, keepdim=True)  # 特征距离

        dist = nn.functional.interpolate(dist, size=x1.shape[2:], mode='bilinear',align_corners=True)
        
        return dist
    
        pred_L = (dist > 1).float()
        # self.pred_L = F.interpolate(self.pred_L, size=self.A.shape[2:], mode='nearest')    

        return pred_L
    
    @staticmethod
    def loss(pred, label):
        label = torch.argmax(label, 1).unsqueeze(1).float()
        return BCL().to('cuda', dtype=torch.float)(pred, label)

    @staticmethod
    def predict(pred):
        prob = (pred > 1.).int()
        return prob
    

class STANetL(nn.Module):
    def __init__(self,in_channels,f_c):
        """
        change detection module:
        feature extractor
        contrastive loss
        """
        super().__init__()
        self.net_f = define_F(in_channels, f_c)
    
    def forward(self, x1, x2=None):
        if x2 == None:
            x1, x2 = x1[:,3,:,:], x1[:,3:,:,:]
        else:
            x1 = x1
            x2 = x2

        feat_A = self.net_f(x1)  # f(A)
        feat_B = self.net_f(x2)   # f(B)

        dist = nn.functional.pairwise_distance(feat_A, feat_B, keepdim=True)  # 特征距离

        dist = nn.functional.interpolate(dist, size=x1.shape[2:], mode='bilinear',align_corners=True)

        pred_L = (self.dist > 1).float()
        # self.pred_L = F.interpolate(self.pred_L, size=self.A.shape[2:], mode='nearest')
        self.pred_L_show = pred_L.long()

        return self.pred_L
    
    @staticmethod
    def loss(pred, label):
        label = torch.argmax(label, 1).unsqueeze(1).float()
        return BCL().to('cuda', dtype=torch.float)(pred, label)
    
    # @staticmethod
    # def loss(pred, label):
    #     label = torch.argmax(label, 1).unsqueeze(1).float()
    #     return BCL().to('cuda', dtype=torch.float)(pred, label)

    @staticmethod
    def predict(pred):
        prob = (pred > 1).int()
        return prob


class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """

    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label[label==255] = 1
        mask = (label != 255).float()
        distance = distance * mask
        pos_num = torch.sum((label==1).float())+0.0001
        neg_num = torch.sum((label==-1).float())+0.0001

        loss_1 = torch.sum((1+label) / 2 * torch.pow(distance, 2)) /pos_num
        loss_2 = torch.sum((1-label) / 2 * mask *
            torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        ) / neg_num
        loss = loss_1 + loss_2
        return loss