import torch
import numpy as np
from work.utils import colour_code_segmentation
import cv2
from datetime import datetime
import os
from common import Metrics, save_numpy_as_csv


def images_prediction(model,data,save_dir,label_info):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("start prediction")
    
    with torch.no_grad():
        model.eval()
        idx = 0

        for _, image in enumerate(data):
            image = image.cuda()
            logits = model(image)
            for logit in logits:
                # logit = logit.squeeze(0)
                # logit = transforms.ToPILImage()(logit)
                logit = torch.argmax(logit, dim=0)
                logit = colour_code_segmentation(np.array(logit.cpu()), label_info)

                logit = cv2.cvtColor(np.uint8(logit), cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_dir+ "_No_" + str(idx) + ".png", logit)
                idx += 1



def evaluation(model,dataloader_eval,args):

    evaluator = Metrics(num_class=args.num_classes)
    with torch.no_grad():
        model.eval()
        p_start = datetime.now()
        num_eval = 0
        for _,(image1, image2, label) in enumerate(dataloader_eval):
            num_eval +=1
            image1 = image1.cuda()
            image2 = image2.cuda()
            label = label.cuda()

            pred = model(image1, image2)

            if hasattr(model, "predict"):
                pred = model.predict(pred)
            elif hasattr(model, "prediction"):
                pred = model.prediction(pred)
            else:
                if (type(pred) == tuple) or (type(pred) == list):
                    pred = pred[args.pred_idx]
            # pred = torch.where(torch.sigmoid(pred) > 0.5, 1, 0)
            # print(pred)
           
            evaluator.add_batch(pred, label)

        evaluator.calc()
        miou = evaluator.Mean_Intersection_over_Union()
        acc = evaluator.Pixel_Accuracy()
        kappa = evaluator.Kappa()
        recall = evaluator.Mean_Recall()
        macro_f1 = evaluator.Macro_F1()
        class_recall = evaluator.Recall()
        class_iou = evaluator.Intersection_over_Union()
        class_precision = evaluator.Class_Precision()
        class_dice = evaluator.Dice()

        args.logger.info("[EVAL] evalution {} images, time: {}".format(num_eval * args.batch_size, datetime.now() - p_start))
        args.logger.info("[METRICS] Acc:{:.4},mIoU:{:.4},recall:{:.4},kappa:{:.4},Macro_f1:{:.4}".format(
            acc,miou,recall,kappa,macro_f1))

        args.logger.info("[METRICS] Class IoU: " + str(np.round(class_iou, 4)))
        args.logger.info("[METRICS] Class Precision: " + str(np.round(class_precision, 4)))
        args.logger.info("[METRICS] Class Recall: " + str(np.round(class_recall, 4)))
        args.logger.info("[METRICS] Class Dice: " + str(np.round(class_dice, 4)))
        
        save_numpy_as_csv(args.metric_path,np.array([args.epoch, args.loss, kappa, miou, acc ,recall,macro_f1]))
        return miou

def test_model(model, data, evaluator,save_file_dir,logger,args):

    with torch.no_grad():
        model.eval()
        evaluator.reset()
        p_start = datetime.now()
        num_eval = 0
        for _,(image,label) in enumerate(data):
            num_eval += 1
            image = image.cuda()
            label = label.cuda()

            pred = model(image)

            # iou, mPA = iou_mpa(pred, label)
            # miou = count_mean_out_nan(iou.cpu())
            #iou = count_mean_out_nan(iou.cpu(),True)

            #label = torch.argmax(label,dim=1)
            evaluator.add_batch(pred,label)

        #now = datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M_%S")
        # miou = np.mean(miou_list).item()
        # mPA = np.mean(mpa_list).item()
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        Kappa = evaluator.Kappa()
        iou = evaluator.Intersection_over_Union()
        f1_score = evaluator.F1_score()
        macro_f1 = evaluator.Macro_F1()

        save_file_dir = save_file_dir.replace('.csv','.txt')
        str1 = "Acc:{:.4},Acc_class:{:.4},mIoU:{:.4},FWIoU:{:.4},kappa:{:.4},Macro_f1:{:.4}".format(
            Acc,Acc_class,mIoU,FWIoU,Kappa,macro_f1)
        # str2 = "iou of classes {:.4},{:.4},{:.4}".format(iou[0],iou[1],iou[2])
        str2 = "iou of classes "
        for ic in iou:
            str2 += '{:.4},'.format(ic)
        # str3 = "f1_score of classes {:.4},{:.4},{:.4}".format(f1_score[0], f1_score[1], f1_score[2])
        str3 = 'f1_score of classes '
        for fc in f1_score:
            str3 += '{:.4},'.format(fc)

        logger.info(str1)
        logger.info(str2)
        logger.info(str3)

        with open(save_file_dir,'w') as f:
            f.writelines(str1+'\n')
            f.writelines(str2+'\n')
            f.writelines(str3+'\n')
        # print(str1)
        # print(str2)
        # print(str3)
        #print("evalution ", num_eval*args.batch_size, "images,time:", datetime.now() - p_start)
        logger.info("evalution {} images,time {}".format(num_eval*args.batch_size,datetime.now() - p_start))


if __name__=="__main__":
    print("work.eval run")

    #TNet = torch.load(args.model_dir)