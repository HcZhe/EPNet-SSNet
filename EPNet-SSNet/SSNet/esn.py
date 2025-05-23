import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, \
    precision_score
import os

CURVE = True
path = './res_dir'
# num = 2
num = 5
# eff = pd.read_csv(path + '/label/data0/eff.csv', header=None, index_col=None).to_numpy()
res = pd.read_csv(path + '/label/data0/res.csv', header=None, index_col=None).to_numpy()
vit = pd.read_csv(path + '/label/data0/vit.csv', header=None, index_col=None).to_numpy()
swin = pd.read_csv(path + '/label/data0/swin.csv', header=None, index_col=None).to_numpy()
# reg = pd.read_csv(path + '/label/data0/reg.csv', header=None, index_col=None).to_numpy()
# dense = pd.read_csv(path + '/label/data0/dense.csv', header=None, index_col=None).to_numpy()
label = pd.read_csv(path + '/label.csv', header=None, index_col=None).to_numpy()
if os.path.exists(path + '/curve') is False:
    os.makedirs(path + '/curve')


def curve(fpr, tpr, pre, rec_, ola):
    au = auc(fpr, tpr)
    apr = auc(rec_, pre)
    pd.DataFrame(np.vstack([fpr, tpr]).T).to_csv(path + '/curve/' + str(ola) + 'auc' + str(au) + '.csv', header=None,
                                                 index=None)
    pd.DataFrame(np.vstack([rec_, pre]).T).to_csv(path + '/curve/' + str(ola) + 'aupr' + str(apr) + '.csv', header=None,
                                                  index=None)


def cal_fpr(scores, ola):
    fpr, tpr, threshold = roc_curve(label, scores)
    pre, rec_, _ = precision_recall_curve(label, scores)
    curve(fpr, tpr, pre, rec_, ola)


def cal_con(l, p):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(l)):
        if p[i]:
            if l[i]:
                TP += 1
            else:
                FP += 1
        else:
            if l[i]:
                FN += 1
            else:
                TN += 1
    print(TP)
    print(FP)
    print(FN)
    print(TN)


if num == 2:
    a, b, c = 1 / 3, 1 / 3, 1 / 3
    score = np.ones(label.shape)
    score = np.hstack([score, eff])
    score = np.hstack([score, res])
    score = np.hstack([score, vit])
    score = np.hstack([score, swin])
    score = np.hstack([score, reg])
    score = np.hstack([score, dense])
    score_res = a * swin + b * reg + c * dense
    score_res = score_res.reshape((-1))
    score = score[:, 1:]
    score_res = score.mean(axis=1)
    if CURVE:
        cal_fpr(score_res, 'ens')
        cal_fpr(eff, 'eff')
        cal_fpr(res, 'res')
        cal_fpr(vit, 'vit')
        cal_fpr(swin, 'swin')
        cal_fpr(reg, 'reg')
        cal_fpr(dense, 'dense')

    fpr, tpr, threshold = roc_curve(label, score_res)
    pre, rec_, _ = precision_recall_curve(label, score_res)
    pred = np.zeros(len(label))
    pred[score_res >= 0.5] = 1
    cal_con(label, pred)
    acc = accuracy_score(label, pred)
    rec = recall_score(label, pred)
    f1 = f1_score(label, pred)
    Pre = precision_score(label, pred)
    au = auc(fpr, tpr)
    apr = auc(rec_, pre)
    # f = open("esn.txt", 'a')
    # f.write("Pre: " + str(round(Pre, 4)) + '\n')
    # f.write("Rec: " + str(round(rec, 4)) + '\n')
    # f.write("ACC: " + str(round(acc, 4)) + '\n')
    # f.write("F1: " + str(round(f1, 4)) + '\n')
    # f.write("AUC: " + str(round(au, 4)) + '\n')
    # f.write("AUPR: " + str(round(apr, 4)) + '\n\n')
    # f.close()
    print('Precision is :{}'.format(Pre))
    print('Recall is :{}'.format(rec))
    print("ACC is: {}".format(acc))
    print("F1 is: {}".format(f1))
    print("AUC is: {}".format(au))
    print('AUPR is :{}'.format(apr))

else:
    score = np.zeros(vit.shape)
    # score += eff
    score += res
    score += vit
    score += swin
    # score += reg
    # score += dense
    # score = score / 6
    score = score / 3
    pred = np.argmax(score, axis=1)
    rec = recall_score(label, pred, average="macro")
    f1 = f1_score(label, pred, average="macro")
    Pre = precision_score(label, pred, average="macro")
    acc = accuracy_score(label, pred)
    print(Pre)
    print(rec)
    print(acc)
    print(f1)
