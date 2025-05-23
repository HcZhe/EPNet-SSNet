import numpy as np
import pandas as pd
# from eff_model import train as effv2_train
# from eff_model import predict as effv2_predict
from resnet_model import train as resnet_train
from resnet_model import predict as resnet_predict
from vit_model import train as vit_train
from vit_model import predict as vit_predict
from swin_model import train as swin_train
from swin_model import predict as swin_predict
# from reg_model import train as reg_train
# from reg_model import predict as reg_predict
# from dense_model import train as dense_train
# from dense_model import predict as dense_predict
from utils import read_data
import os
import torch


# 创建需要的目录结构
def mk_dir():
    if os.path.exists('./res_dir/label/data') is False:
        os.makedirs('./res_dir/label/data')
    if os.path.exists('./res_dir/train_res/data') is False:
        os.makedirs('./res_dir/train_res/data')
    if os.path.exists('./res_dir/weights/data') is False:
        os.makedirs('./res_dir/weights/data')
    if os.path.exists('./res_dir/best_res') is False:
        os.makedirs('./res_dir/best_res')


category_to_label_map = {
    'amphibious': 0,
    'fly': 1,
    'insect': 2,
    'land': 3,
    'sea': 4
}

val = False  # 设置是否进行验证
TRAIN = False  # 设置是否进行训练
PREDICT = True  # 设置是否进行预测
PRE_train = True  # 设置是否使用预训练模型
n_class = 5  # 设置分类数目
ep = 5  # 设置训练周期
device = 'cuda:0'  # 设置使用的设备

# 调用函数读取训练数据
train_data_path = '/root/autodl-tmp/SwinT-Classif/CoCOD8K_2/train/image/'
# 调用函数读取测试数据，不进行分割
test_data_path = '/root/autodl-tmp/SwinT-Classif/transAndCod/test/image/'

print(torch.version.cuda)
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

if __name__ == '__main__':
    for i in range(0, 1):
        mk_dir()  # 调用函数创建目录
        train_paths, val_paths, train_labels, val_labels = read_data(train_data_path, category_to_label_map, k=0.2,
                                                                     seed=5, is_test=False)
        test_paths, test_labels = read_data(test_data_path, category_to_label_map, is_test=True)  # 如果设置为训练
        if TRAIN:
            # effv2_train(train, train_label, classes=n_class, device=device, val=False, data=i, epochs=ep, init=PRE_train)
            resnet_train(train_paths, train_labels, val_paths, val_labels , classes=n_class, device=device, val=val, epochs=ep, init=True)
            # vit_train(train_paths, train_labels, val_paths, val_labels, classes=n_class, device=device, val=val, epochs=ep, init=PRE_train)
            # swin_train(train_paths, train_labels, val_paths, val_labels, classes=n_class, device=device, val=val,
                       # epochs=ep, init=PRE_train)
            # reg_train(train, train_label, classes=n_class, device=device, val=False, data=i, epochs=ep, init=PRE_train)
            # dense_train(train, train_label, classes=n_class, device=device, val=False, data=i, epochs=ep, init=PRE_train)
            print('train over')
        # 如果设置为预测
        if PREDICT:
            # score1 = effv2_predict(test, test_label, num_class=n_class, data=i)
            score2 = resnet_predict(test_paths, test_labels, num_class=n_class)
            # score3 = vit_predict(test_paths, test_labels, num_class=n_class)
            # score4 = swin_predict(test_paths, test_labels, num_class=n_class)
            # score5 = reg_predict(test, test_label, num_class=n_class, data=i)
            # score6 = dense_predict(test, test_label, num_class=n_class, data=i)
            pd.DataFrame(np.array(test_labels)).to_csv('./res_dir/label.csv', header=None, index=None)
            print('predict over')
