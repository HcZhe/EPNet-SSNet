# `EPNet-SSNet`

```
This repository contains the official PyTorch implementation of the paper:

"EPNet and SSNet: Towards Exploiting Pose for RGB-D Human Action Recognition"  
by Chenzhe Hu, Xinying Wang, Jin Tang, and Junsong Yuan  
Published in IEEE Transactions on Image Processing (TIP), 2023.
```

## `📂 Directory Structure`

```
EPNet-SSNet/
├── backbone/
├── datasets/
├── networks/
├── results/
├── scripts/
├── utils/
├── main.py
├── options.py
├── requirements.txt
├── README.md
```

## `📦 Requirements`

```
Python 3.8  
PyTorch 1.8.1  
torchvision 0.9.1  
Other requirements listed in requirements.txt
```

### Installation

```bash
git clone https://github.com/HcZhe/EPNet-SSNet.git
cd EPNet-SSNet
pip install -r requirements.txt
```

## `📁 Datasets`

```
NTU RGB+D 60  
NTU RGB+D 120  
Northwestern-UCLA  
```

### Dataset Preparation

```
- Download datasets manually from the official sources.
- Extract to the datasets/ directory.
- Follow preprocessing scripts provided in `datasets/` to prepare training and test data.
```

## `🚀 Training`

### EPNet

```bash
python main.py --config scripts/EPNet_train.yaml
```

### SSNet

```bash
python main.py --config scripts/SSNet_train.yaml
```

## `🔍 Testing`

```bash
python main.py --config scripts/EPNet_test.yaml
python main.py --config scripts/SSNet_test.yaml
```

## `📊 Results`

```
Results will be saved in the results/ directory.  
Logs and model checkpoints are also stored in the same directory.
```



## `📬 Contact`

```
For any questions or suggestions, feel free to contact the author via GitHub issues.
```
