#调用官方库及第三方库
import torch
import numpy as np
#from tensorboardX import SummaryWriter
import datetime
from torch import optim
import os

#基础功能
from work.utils import get_params
from work.val import evaluation
from work.predict import test, test_last
from work.utils import get_scheduler


def loss(logits,labels):
    # return nn.BCEWithLogitsLoss()(logits, lab)

    if logits.shape == labels.shape:
        labels = torch.argmax(labels,dim=1)
    elif len(labels.shape) == 3:
        labels = labels
    else:
        assert "pred.shape not match label.shape"
    #logits = F.softmax(logits,dim=1)
    return torch.nn.CrossEntropyLoss()(logits,labels)


def dice_loss(logits,targets,smooth=1e-5):
    # logic.shape=[C,N,H,W],target.shape=[C,N,H,W]&[C,1,H,W]
    if logits.shape == targets.shape:
        targets = targets.type(torch.float32)
    elif targets.shape[1] == 1:
        targets = targets.type(torch.int64)
        targets = torch.zeros_like(logits).scatter_(dim=1, index=targets, src=torch.tensor(1.0)).type(torch.float32)

    else:
        assert "pred.shape not match label.shape"

    outputs = torch.nn.functional.softmax(logits,dim=1).type(torch.float32)
    #outputs = logits
    #targets=torch.zeros_like(logits).scatter_(dim=1,index=targets,src=torch(1.0))

    inter = outputs*targets
    dice = 1 - ((2 * inter.sum(dim=(2, 3)) + smooth) / (outputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth))
    return dice.mean().item()

def train(model, dataloader_train, dataloader_eval, dataloader_test, args):

    args.logger.info("start train")
    #optimizer = optim.SGD(TNet.parameters(), opt_params)
    # optimizer = optim.SGD(
    #     params=[
    #         {'params': get_params(model, key='1x'), 'lr': 5e-4},
    #         {'params': get_params(model, key='10x'), 'lr': 5e-3}
    #     ],
    #     momentum = 0.9 , weight_decay = 1e-4)
    optimizer = optim.Adam(model.parameters(),lr= 5e-4, betas=(0.9, 0.999))
    # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult = 1, eta_min = 0, last_epoch = -1, verbose = False)
    
    max_itr = args.iters * len(dataloader_train)

    lr_step = get_scheduler(optimizer, max_itr, 'step')
    max_miou = 0.
    best_iter = 0
    #early_stopping = Early_stopping(eps=2e-5,llen=10)
    #criterion = SegmentationLosses(weight=None,cuda=True).build_loss("ce")

    for epoch in range(args.iters):
        # if test:
        #     test_model(model, dataloader_eval, evaluator,args)
        #     images_prediction(model, dataloader_pred, save_dir=save_folder, label_info=label_info)
        #     break
        # np.random.seed(53113)
        # torch.manual_seed(53113)
    #
        now = datetime.datetime.now()
        model.train()
        loss_record = []
        #evaluator.reset()

        for _,(image1, image2, label) in enumerate(dataloader_train):

            #optimizer = adjust_lr(optimizer, epoch*iter, max_itr)
            
            image1 = image1.cuda()
            image2 = image2.cuda()
            label = label.cuda()
            
            pred = model(image1, image2)
            
            if hasattr(model, "loss"):
                reduced_loss = model.loss(pred, label)
            else:
                if (type(pred) == tuple) or (type(pred) == list):
                    pred = pred[args.pred_idx]
                reduced_loss = loss(pred, label)

            optimizer.zero_grad()  # 梯度清零
            reduced_loss.backward()  # 计算梯度
            optimizer.step()
            lr_step.step()

            loss_record.append(reduced_loss.item())

        loss_tm = np.mean(loss_record)

        args.epoch = epoch + 1.0
        args.loss = loss_tm

        args.logger.info("[TRAIN] iter:{}/{}, learning rate:{:.6}, loss:{:.6}".format(epoch+1, args.iters, optimizer.param_groups[0]['lr'], loss_tm))
        miou_eval = evaluation(model,dataloader_eval,args)

        if (epoch+1) % 10 ==0:
             torch.save(model.state_dict(), os.path.join(args.save_dir, "iter_{}.pth".format(epoch+1)))
        
        # if miou_eval > max(0.7, max_miou):
        #     torch.save(model.state_dict(), args.best_model_path)

        if miou_eval > max(0.5, max_miou):
            torch.save(model.state_dict(), args.best_model_path)
            max_miou = miou_eval
            best_iter = epoch+1
        args.logger.info("[TRAIN] train time is {:.2f}, best iter {}, max mIoU {:.4f}".format((datetime.datetime.now() - now).total_seconds(), best_iter, max_miou))
        #print("+++++++++++++++++++++++++++++++time:", datetime.datetime.now() - now)
        # early_stopping.add_data(loss_tm)
        # if early_stopping.en_stop(force=True):
        #     break
        # test_model(model, dataloader_eval, evaluator)
        # images_prediction(model, dataloader_pred, save_dir=save_folder, label_info=label_info)
    
    test(model, dataloader_test, args)
    # last_path = os.path.join(args.save_dir, "iter_{}.pth".format(args.iters))
    # test_last(model, dataloader_test, args, last_path)
    
    # torch.save(TNet.state_dict(),save_folder+"last.path")




if __name__ == "__main__":
    print("work.train run")




