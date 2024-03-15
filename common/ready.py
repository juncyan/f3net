import datetime
import os
import argparse
from .csver import writer_csv
from .logger import load_logger

__all__ = ['Args', 'setup_train']

class Args():
    def __init__(self, dst_dir, model_name):
        # [epoch, loss, acc, miou, mdice,kappa,macro_f1]
        demo_predict_data_headers = ["epoch", "loss", "Kappa", "miou", "acc", "recall", 'Macro_f1']
        self.num_classes = 2
        self.batch_size = 8
        self.iters = 0
        self.pred_idx = 0
        self.data_name = ""
        self.model_name = model_name
        
        self.save_dir = os.path.join(dst_dir, r"{}_{}".format(model_name,datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")))
        # self.save_predict = os.path.join(self.save_dir , "predict")
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # if not os.path.exists(self.save_predict):
        #     os.makedirs(self.save_predict)

        self.best_model_path = os.path.join(self.save_dir, "{}_best.pth".format(model_name))
        log_path = os.path.join(self.save_dir, "train_{}.log".format(model_name))
        self.metric_path = os.path.join(self.save_dir, "{}_metrics.csv".format(model_name))
        print("log save at {}, metric save at {}, weight save at {}".format(log_path, self.metric_path, self.best_model_path))
        self.epoch = 0
        self.loss = 0
        self.logger = load_logger(log_path)
        self.logger.info("log save at {}, metric save at {}, weight save at {}".format(log_path, self.metric_path, self.best_model_path))
        writer_csv(self.metric_path, headers=demo_predict_data_headers)


def setup_train(args, model):
    model_name = model.__str__().split("(")[0]
    #demo_predict_data_headers = ["void", "build", "forest", "river", "miou", "mpa"]
    demo_predict_data_headers = ["mIoU","acc","Kappa",'Macro_f1']
    time_flag = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H")

    save_model_dir = args.result_dir + model_name + "/model_best.path"
    save_folder = args.result_dir + model_name + "/" + time_flag + "/"
    save_log_dir = save_folder + model_name + ".log"
    #log_dir = args.log_dir + model_name + "/" + time_flag + "/"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # if not os.path.exists(save_model_dir):
    #     os.makedirs(save_model_dir)
    paramters_txt = save_folder + model_name+"_metrics.csv"
    writer_csv(paramters_txt, headers=demo_predict_data_headers)

    return save_folder, paramters_txt, save_model_dir, save_log_dir

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Overfitting Test')
    # model
    parser.add_argument('--model', type=str, default='fcn32s',
                        choices=['fcn32s', 'fcn16s', 'fcn8s', 'fcn', 'psp',
                        'deeplabv3', 'danet', 'denseaspp', 'bisenet', 'encnet',
                        'dunet', 'icnet', 'enet', 'ocnet'],
                        help='model name (default: fcn32s)')
    parser.add_argument('--backbone', type=str, default='vgg16',
                        choices=['vgg16', 'resnet18', 'resnet50', 'resnet101',
                        'resnet152', 'densenet121', '161', '169', '201'],
                        help='backbone name (default: vgg16)')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys',
                        'sbu'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    args = parser.parse_args()
    args.device = 'gpu'
    print(args)
    return args