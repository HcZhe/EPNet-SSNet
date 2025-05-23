import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, \
    precision_score, confusion_matrix
import matplotlib.pyplot as plt

# 从一个文本文件中读取数据，并根据数据的类别将其分类
def read_one_data(path_to_data):
    # 定义类别与标签ID的映射
    category_to_label = {
        'amphibious': 0,
        'fly': 1,
        'insect': 2,
        'land': 3,
        'sea': 4
    }

    # 用于存储所有图像路径和对应标签的列表
    paths = []
    labels = []

    # 遍历数据文件夹
    for category, label in category_to_label.items():
        category_path = os.path.join(path_to_data, category)  # 获取每个类别的路径
        images = os.listdir(category_path)  # 列出文件夹中的所有图像文件

        # 为当前类别的每个图像文件添加路径和标签
        for image in images:
            image_path = os.path.join(category_path, image)
            paths.append(image_path)
            labels.append(label)

    return paths, labels


# 从指定的文件路径 path_file 中读取数据，并将其分为训练集和验证集
# 同时为每个图像分配相应的标签
"""
path_file：包含图像路径的文件路径。
k：用于分割训练集和验证集的比例（通常是一个介于0到1之间的小数值，表示验证集占总数据的比例）。
seed：随机数生成器的种子，用于确保数据分割的可重复性。
n_class：数据集中的类别数，可以是2表示二分类问题，或大于2表示多分类问题。"""


# def read_data(path_file, k, seed=5, n_class=5):
#     file = open(path_file)
#     path = file.readlines()
#     train_path = []
#     train_label = []
#     val_path = []
#     val_label = []

#     # 定义类别与标签ID的映射
#     category_to_label = {
#         'amphibious': 0,
#         'fly': 1,
#         'insect': 2,
#         'land': 3,
#         'sea': 4
#     }

#     # 填充每个类别的列表
#     for i in path:
#         p = i.split('\n')[0]
#         category = p.split('/')[1]  # 假设类别名称是路径中的第二个元素
#         if category in category_to_label:
#             label = category_to_label[category]
#             data_list = globals()[f"{category}_data"]  # 使用 globals() 和 f-string 动态访问变量
#             data_list.append(p)

#     # 对每个类别的数据集进行分割
#     for category, data_list in globals().items():  # 使用 globals() 遍历全局变量
#         if isinstance(data_list, list) and data_list:  # 确保这是列表且非空
#             train_data, val_data = train_test_split(data_list, test_size=k, random_state=seed)
#             train_path.extend(train_data)
#             train_label.extend([label] * len(train_data))  # 扩展标签列表
#             val_path.extend(val_data)
#             val_label.extend([label] * len(val_data))  # 扩展标签列表

#     return train_path, train_label, val_path, val_label

# def read_data(data_path, k, seed=5, n_class=5):
#     # 定义类别与标签ID的映射
#     category_to_label = {
#         'amphibious': 0,
#         'fly': 1,
#         'insect': 2,
#         'land': 3,
#         'sea': 4
#     }

#     # 用于存储所有图像路径和对应标签的列表
#     paths = []
#     labels = []

#     # 遍历数据文件夹中的每个类别
#     for category, label in category_to_label.items():
#         category_path = os.path.join(data_path, category)  # 获取每个类别的路径
#         if os.path.isdir(category_path):  # 检查路径是否为目录
#             images = os.listdir(category_path)  # 列出文件夹中的所有文件
#             for image in images:
#                 full_path = os.path.join(category_path, image)
#                 if os.path.isfile(full_path):  # 确保它是一个文件
#                     paths.append(full_path)
#                     labels.append(label)

#     # 将数据分为训练集和验证集
#     train_path, val_path, train_label, val_label = train_test_split(
#         paths, labels, test_size=k, random_state=seed
#     )

#     return train_path, train_label, val_path, val_label
def read_data(data_path, category_to_label_map, k=0.2, seed=5, is_test=False):
    """
    读取指定路径下的数据，将其分为训练集和验证集或返回全部数据（如果是测试集）。

    参数:
    data_path (str): 数据集的根路径。
    category_to_label_map (dict): 类别名称到标签的映射。
    k (float): 用于分割训练集和验证集的比例。
    seed (int): 随机数生成器的种子，用于确保数据分割的可重复性。
    is_test (bool): 是否为测试集，如果是，则不进行分割。

    返回:
    list: 图像路径的列表。
    list: 对应的标签列表。
    """

    # 用于存储所有图像路径和对应标签的列表
    paths = []
    labels = []

    # 遍历数据文件夹中的每个类别
    for category, label in category_to_label_map.items():
        category_path = os.path.join(data_path, category)  # 获取每个类别的路径
        if os.path.isdir(category_path):  # 检查路径是否为目录
            images = os.listdir(category_path)  # 列出文件夹中的所有文件
            # print(f"Reading {len(images)} images from {category_path}...")  # 打印每个类别读取的图像数量
            
            for image in images:
                full_path = os.path.join(category_path, image)
                if os.path.isfile(full_path):  # 确保它是一个文件
                    paths.append(full_path)
                    labels.append(label)
                else:
                    print(f"Skipped non-image file: {full_path}")  # 打印跳过的非图像文件
    # 如果是测试集，不进行分割
    if not is_test:
        # 将数据分为训练集和验证集
        train_path, val_path, train_label, val_label = train_test_split(
            paths, labels, test_size=k, random_state=seed
        )
        # print("Sample train paths:", train_path[:5])  # 打印样本训练路径
        # print("Sample validation paths:", val_path[:5])  # 打印样本验证路径
        return train_path, val_path, train_label, val_label
    else:
        # 对于测试集，直接返回所有路径和标签
        return paths, labels


# 在一个epoch中训练模型。它计算并累加损失和准确率，并在每个数据批次后更新模型参数
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    # 将模型设置为训练模式。这将启用如 Dropout 等只在训练期间使用的操作
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    # 初始化两个用于累积损失和累积正确预测数量的张量
    # 使用 torch.zeros 创建，并使用 .to(device) 将它们移动到指定的设备（CPU或GPU）
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    """在 PyTorch 中，torch.zeros(1) 是用来创建一个大小为 1 的零张量（tensor）的函数调用。
    这个张量的数据类型默认是 float32，除非你指定了其他数据类型。
    这里的 1 指的是张量将会是一个一维数组，它只有一个元素，且该元素的值为 0"""

    # 在每次前向传播之前，清除优化器中的所有梯度
    optimizer.zero_grad()
    """优化器（Optimizer）在机器学习中是一个用于在训练过程中更新模型参数的算法。
    它根据损失函数（Loss Function）相对于模型参数的梯度来调整参数，
    目的是最小化损失函数，从而提高模型的预测性能"""
    # 累计epoch中的样本数量
    sample_num = 0
    # 使用 tqdm 库包装数据加载器，以便在训练过程中显示进度条
    data_loader = tqdm(data_loader)
    # 开始遍历数据加载器的每一批次数据。step 是批次的索引，data 包含图像和标签
    for step, data in enumerate(data_loader):
        # data 是从数据加载器（DataLoader）中获取的批次数据，提供了图像和标签的批次
        images, labels = data
        # image是一个tensor张量 shape[0]表示batchsize（样本数量） N H W C
        sample_num += images.shape[0]
        # 使用模型对当前批次的图像进行预测，并将图像数据移动到指定的设备
        pred = model(images.to(device))
        # 从预测结果中获取最可能的类别 dim=1代表在”行“这个维度进行操作
        """pred 是模型对一批图像的预测输出，
        它通常是一个二维张量，其中每一行代表一个图像的预测结果，
        每一列代表一个类别的得分（或概率）
        torch.max(pred, dim=1) 返回两个值：最大值和最大值的索引,[1] 则是用来获取最大值的索引，即预测的类别"""
        pred_classes = torch.max(pred, dim=1)[1]
        # 从预测结果中获取最可能的类别
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        # 确保标签是长整型，以匹配损失函数的输入要求
        labels = labels.long()
        # 计算预测和真实标签之间的损失
        loss = loss_function(pred, labels.to(device))
        # 执行反向传播，计算损失相对于模型参数的梯度
        loss.backward()
        # 累加损失，使用 .detach() 方法创建损失的副本，防止在损失计算图中引入额外的依赖
        accu_loss += loss.detach()
        """在 PyTorch 中，当你执行 loss.backward() 进行反向传播时，所有的计算都会构建出一个计算图，
        以便计算梯度。loss.detach() 方法是从一个张量中创建一个新的张量，该新张量共享原始数据但没有任何计算图的依赖，即它不会参与梯度计算。
        这么做的原因是：在累加损失时，我们只想累加损失的值，而不是累加损失的梯度。因为梯度累加会导致数值问题，如梯度爆炸或消失。
        使用 .detach() 可以确保累加的是损失值，而不会影响原始损失张量的梯度计算图。"""
        # 更新进度条的描述，显示当前epoch的损失和准确率
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        # 检查损失是否为有限数，如果不是，打印警告并终止训练
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        # 使用优化器根据计算出的梯度更新模型的参数
        optimizer.step()
        # 清除优化器中的梯度，为下一次迭代做准备
        optimizer.zero_grad()
    # 函数返回epoch的平均损失和准确率。使用 .item() 将张量转换为Python数值以便返回
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


# 在一个epoch中评估模型性能。它计算验证集上的损失和准确率
# @torch.no_grad() 是一个装饰器，用于指定在函数内部执行的所有操作都不会计算或存储梯度，这在模型评估时是常用的，因为评估时不需要更新模型参数。
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        labels = labels.long()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


# 计算模型的多种评价指标，如准确率、召回率、F1分数、AUC和AUPR
# 还输出了这些指标的结果，并将其写入到 res_dir/res.txt 文件中
@torch.no_grad()
def cal_m(model, data_loader, device, alo='def'):
    model.eval()
    all_labels = np.array([])
    all_preds = np.array([])

    # 遍历数据加载器中的所有数据
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds = np.concatenate((all_preds, preds.cpu().numpy()))
        all_labels = np.concatenate((all_labels, labels.cpu().numpy()))

    # 计算各种评价指标
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    # 输出评价指标
    print(f'Accuracy: {accuracy}')
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')
    print(f'F1 Score: {f1}')
    print('Confusion Matrix:')
    print(cm)

    # 将评价指标写入文件
    with open("./res_dir/res.txt", 'a') as f:
        f.write(f"{alo}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm, separator=', ') + '\n\n')

    return (accuracy, recall, precision, f1), cm



# num_classes = 5
# @torch.no_grad()
# def cal_m(model, data_loader, device, alo='def'):
#     model.eval()
#     all_labels = np.array([])
#     all_probs = []  # 用列表收集每个类别的预测概率

#     for images, labels in tqdm(data_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()  # 获取每个类别的预测概率
#         all_probs.append(probs)
#         all_labels = np.concatenate((all_labels, labels.cpu().numpy()))

#     # 将预测概率转换为模型输出的形式
#     all_probs = np.vstack(all_probs)

#     # 计算评价指标
#     accuracy = accuracy_score(all_labels, np.argmax(all_probs, axis=1))

#     # 计算每个类别的PR曲线和ROC曲线
#     precision = dict()
#     recall = dict()
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()

#     for i in range(num_classes):  # 假设变量num_classes保存了类别数目
#         precision[i], recall[i], _ = precision_recall_curve(np.equal(all_labels, i), all_probs[:, i])
#         fpr[i], tpr[i], _ = roc_curve(np.equal(all_labels, i), all_probs[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])

#     # 绘制PR图
#     plt.figure()
#     class_labels = ['amphibious', 'fly', 'insect', 'land', 'sea']
#     for i in range(num_classes):
#           plt.plot(recall[i], precision[i], lw=2, label='{}: AP={:.2f}'.format(class_labels[i], auc(recall[i], precision[i])))
#         # plt.plot(recall[i], precision[i], lw=2, label='Class {}: AP={:.2f}'.format(i, auc(recall[i], precision[i])))
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     # plt.title('Precision-Recall Curve')
#     plt.title('ResNet PR')
#     plt.legend(loc="upper right")
#     plt.savefig('./res_dir/pic/pr_curve_{}.png'.format(alo))
#     plt.close()

#     # 绘制ROC曲线图
#     plt.figure()
#     for i in range(num_classes):
#         plt.plot(fpr[i], tpr[i], lw=2, label='{}: ROC AUC={:.2f}'.format(class_labels[i], roc_auc[i]))
#         # plt.plot(fpr[i], tpr[i], lw=2, label='Class {}: ROC AUC={:.2f}'.format(i, roc_auc[i]))
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     # plt.title('Receiver Operating Characteristic')
#     plt.title('ResNet ROC')
#     plt.legend(loc="lower right")
#     plt.savefig('./res_dir/pic/roc_curve_{}.png'.format(alo))
#     plt.close()

#     return accuracy
