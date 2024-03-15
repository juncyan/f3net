import numpy as np
import torch


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return IoU

    def Mean_Intersection_over_Union(self):
        # MIoU = np.diag(self.confusion_matrix) / (
        #             np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
        #             np.diag(self.confusion_matrix))
        # MIoU = np.nanmean(MIoU)
        IoU = self.Intersection_over_Union()
        MIoU = np.nanmean(IoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Kappa(self):
        diag = np.diag(self.confusion_matrix)
        clo_sum = np.sum(self.confusion_matrix,axis=0)
        row_sum = np.sum(self.confusion_matrix,axis=1)
        num = np.sum(self.confusion_matrix)
        P0 = np.sum(diag)/num
        Pe = np.sum(row_sum*clo_sum)/(num*num)
        return (P0-Pe)/(1-Pe)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    #def add_batch(self, gt_image, pre_image):
    def add_batch(self, pred, lab):
        if len(lab.shape)==4:
            lab = torch.argmax(lab, dim=1)
        gt_image = np.array(lab.cpu())
        pre_image = torch.argmax(pred,dim=1)
        pre_image = np.array(pre_image.cpu())
        # print(gt_image.shape)
        # print(pre_image.shape)
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def F1_score(self,belta=1):
        TP = np.diag(self.confusion_matrix)
        RealN = np.sum(self.confusion_matrix,axis=0) # TP+FN
        RealP = np.sum(self.confusion_matrix,axis=1) # TP+FP
        precision = TP/RealP
        recall = TP/RealN
        f1_score = (1+belta*belta)*precision*recall/(belta*belta*precision+recall)
        return f1_score

    def Macro_F1(self,belta=1):
        return np.nanmean(self.F1_score(belta))

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


if __name__ == "__main__":
    print("TNet.Metrics run")
    x = [0.83611815, 0.51126306, 0.69839933 ,0.75191475]
    print(np.mean(x))




