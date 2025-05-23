import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.Network_Res2Net_GRA_NCD import Network
from utils.data_val import test_dataset
import imageio
from datetime import datetime #hcz_Add
import logging #hcz_Add


# 设置日志
logging.basicConfig(
    filename='./snapshot/SINet_V2/log.log',  # 指定日志文件的存储路径和文件名
    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',  # 定义日志的格式
    level=logging.INFO,  # 设置日志级别为INFO
    filemode='a',  # 设置文件模式为追加模式
    datefmt='%Y-%m-%d %H:%M:%S'  # 设置时间戳格式，24小时制时间显示
)

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshot/SINet_V2/Net_epoch_best.pth')
opt = parser.parse_args()

for _data_name in ['CAMO', 'COD10K', 'CHAMELEON']:
    data_path = './Dataset/TestDataset/{}/'.format(_data_name)
    save_path = './res/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    model = Network()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        #image, gt, name = test_loader.load_data()
        image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res5, res4, res3, res2 = model(image)
        res = res2
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> ')
        # 此为原代码，但是版本太老，故修改misc.imsave(save_path+name, res) 
        # 假设 res 是你的图像数据，范围在 [0, 1]
        res_uint8 = (res * 255).astype(np.uint8)  # 将浮点数转换为 uint8
        imageio.imwrite(save_path+name, res_uint8)
        
    logging.info(f'Testing for {_data_name} completed at {datetime.now()}')

logging.info('All tests completed successfully.')
        
