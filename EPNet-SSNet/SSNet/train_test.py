import numpy as np
from eff_model import train as effv2_train
from eff_model import predict as effv2_predict
from resnet_model import train as resnet_train
from resnet_model import predict as resnet_predict
from vit_model import train as vit_train
from vit_model import predict as vit_predict
from swin_model import train as swin_train
from swin_model import predict as swin_predict
from reg_model import train as reg_train
from reg_model import predict as reg_predict
from dense_model import train as dense_train
from dense_model import predict as dense_predict
from utils import read_data

val = False
TRAIN = True
PREDICT = True
if __name__ == '__main__':
    for lr in [0.1, 0.01, 0.001, 0.0001]:
        data_path = './Dataset/test/image/'
        train, train_label, test, test_label = read_data(data_path, 0.2)
        # effv2_train(train, train_label, classes=5, val=False, data=i, lr=lr, epochs=100)
        # label1, score1 = effv2_predict(test, test_label, data=i)
        # print('lr = {}'.format(lr))

        # resnet_train(train, train_label, classes=2, val=False, data=i)
        # label2, score2 = resnet_predict(test, test_label, data=i)
        #
        # vit_train(train, train_label, classes=2, val=False, data=4, lr=lr, epochs=100)
        # label3, score3 = vit_predict(test, test_label, data=4)
        #
        swin_train(train, train_label, classes=5, val=False)
        label4, score4 = swin_predict(test, test_label)
        #
        # reg_train(train, train_label, classes=2, val=False, data=i)
        # label5, score5 = reg_predict(test, test_label, data=i)
        #
        # dense_train(train, train_label, classes=2, val=False, data=i)
        # label6, score6 = dense_predict(test, test_label, data=i)
