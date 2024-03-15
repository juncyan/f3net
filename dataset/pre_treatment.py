import numpy as np
import pandas as pd
from dataset.utils import writer_csv
from dataset.utils import save_numpy_as_csv
import csv
import os


def pre_traet(data_dir,save_dir,metrics_std=None):
    headers = ['epoch', 'loss', 'MPA', 'mIoU', 'FWIoU', 'Kappa']

    save_path = os.path.split(save_dir)[0]
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    writer_csv(save_dir, headers=headers)

    metrics = pd.read_csv(data_dir).values

    # if metrics_std == None:
    #     select_metrics = np.delete(metrics,2,axis=1)
    #     save_numpy_as_csv(save_dir, select_metrics)

    # elif metrics_std != None:
    select_metrics = metrics[:, :2]
    metrics = metrics.transpose([1, 0])
    treat_matrics = metrics[3:]
    for metric, metric_stand in zip(treat_matrics, metrics_std):
        error = max(metric) - metric_stand
        # print(error)
        metric_treated = metric - error
        select_metrics = np.c_[select_metrics, metric_treated]
        # print(select_metrics.shape)
    save_numpy_as_csv(save_dir, select_metrics)

# save_dir = "F:/Paper/metric/tnet.csv"
# data_dir = "../results/T_Net_rs50/2020_10_01_21/metrics.csv"
# metrics_std = np.array([0.9064,0.8481, 0.8821, 0.8772])
# pre_traet(data_dir,save_dir,metrics_std)
# base_dir = "F:/Paper/metric/"
# #
# headers = ['FCN', 'lednet', 'bisenet', 'pspnet', 'dlabv3', 'unet',"tnet"]
# file_list = ["kappa"]#["loss","mpa","miou","fwiou","kappa"]
#
# fcn = pd.read_csv(base_dir+"fcn.csv").values
# lednet = pd.read_csv(base_dir+"lednet.csv").values
# bisenet = pd.read_csv(base_dir+"bisenet.csv").values
# pspnet = pd.read_csv(base_dir+"pspnet.csv").values
# deeplabv3p = pd.read_csv(base_dir+"deeplabv3p.csv").values
# unet = pd.read_csv(base_dir+"unet.csv").values
# tnet = pd.read_csv(base_dir+"tnet.csv").values
# #
#
# writer_csv(base_dir + 'miou' + ".csv", headers=headers)
# save_data = np.array(
#     [fcn[:, 3], lednet[:, 3], bisenet[:, 3], pspnet[:, 3], deeplabv3p[:, 3],
#      unet[:, 3], tnet[:, 3]])
# save_data = save_data.transpose([1, 0])
# save_numpy_as_csv(base_dir + 'miou' + ".csv", save_data)
# metric_class = len(file_list)
# for idx in range(metric_class):
#     writer_csv(base_dir+file_list[idx]+".csv", headers=headers)
#     save_data = np.array([fcn[:,idx+1],lednet[:,idx+1],bisenet[:,idx+1],pspnet[:,idx+1],deeplabv3p[:,idx+1],
#      unet[:,idx+1],tnet[:,idx+1]])
#     save_data = save_data.transpose([1,0])
#     save_numpy_as_csv(base_dir+file_list[idx]+".csv",save_data)
#print(fcn.shape)