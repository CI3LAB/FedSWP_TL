# 🌟 FedSWP-TL

## 📌 Description (项目概述)
The implementation of federated smart work package with triplet loss (FedSWP-TL), 
a lightweight and adaptive FedSWP-TL workflow to address the need for efficient fatigue monitoring of crane operators.

### 📂 The Files (项目结构)
```
.
├── scripts/                         # core scripts(核心代码)
│   ├── main.py                      # the main function(主处理逻辑)
│   ├── data_processing_*.py         # processes multiple datasets: YawDD, DROZY, and ConPPMF(数据处理模块) 
│   ├── trainer.py                   # trainer (训练逻辑)
│   ├── swp0.py                      # raw main like trainer(训练逻辑)
│   ├── embedding.py                 # network and model architecture (模型结构)
│   ├── triplet_loss.py              # network triplet loss (损失函数)
│   └── utils.py                     # tools for data processing, evaluations, etc (工具集成)
├── models/                          # trained models (单元测试)
├── data/                            # data (数据文件)
├── examples/                        # examples(示例代码)
├── requirements.txt                 # requirements(依赖列表)
└── README.md                        # description
```

### 🚀  Datasets
**Title: YawDD** 
short for “Yawing Detection Dataset,” comprised two datasets featuring videos captured at a resolution of 640 × 480 pixels with 24-bit actual color (RGB) from car operations in 2014. It was identical to alerting fatigue of the crane operator. These videos depicted a range of facial expressions, including regular, talking/singing, and yawning at 30 frames per second (FPS). The first dataset, obtained from the front mirror perspective, comprised 270 videos featuring 90 subjects (47 males and 43 females). The second dataset, captured from the dashboard viewpoint, involved 29 subjects (16 males and 13 females).   
**Title: DROZY**  
The ULg Multimodality Drowsiness Database (Massoz et al., 2016), stands out as the most widely utilized dataset for monitoring mental fatigue. Collected over two days in a controlled lab setting, it encompassed multiple modalities, such as EEG (Fz, Pz, Cz, C3, and C4), EOG, ECG, EMG, camera videos and reaction times via the administration of the psychomotor vigilance test (PVT), the application of polysomnography (PSG) electrodes, and the KSS assessment of sleepiness levels. The study cohort included 14 healthy participants, consisting of 3 males and 11 females, with an average age of 22.7 years (±2.3 SD).  
**Title: ConPPFM**  
Creator: Zeng, Jianhuan  
Department/Faculty: Department of Civil Engineering  
Data Owner: Mr Li Xiao, Assistant Professor; email: shell.x.li@hku.hk  
Data Custodian:  Ms Zeng Jianhuan, PhD Student; email: joy2023@connect.hku.hk; 92196237  
HKU staff members: via e-Form (e.g. CF-XXX);  
Non-HKU members: via email addressing to data steward (shell.x.li@hku.hk).
Persistent identifier
Subject
Funders: HKU CI3 Lab  
Rights
Access information: email joy2023@connect.hku.hk for datasets.  
Dates: 2024/10/25
File names: confm_cr_202410
File format: mp4, csv  
Versions: v20241001  


## 🏃 Get Started (快速开始)

### 🛠 Preparations (前置要求)

Two virtual machines are employed in the experimental environment, running on a Linux Ubuntu 20.04 system, to execute the training and evaluation process. The computer specifications and development package details for this experiment are provided below:
- Hardware: 20 cores, 64 GB of RAM, 256 GB SSD, and 2 TB HDD
- Network: Each node can accommodate up to three open ports
- Architecture: AMD64
- Hosting Platform for containers: Docker 18.01.0
- CPU: Intel R XeR (R) E5-2640 v4@ 2.40 GHz (20 CPUs)
- GPU: NVIDIA GeForce GTX 1080
- Deep learning framework: Python 3.8+, PyTorch 2.0+, CUDA 11.7

### 🔧 Parameters (配置选项)
1. dataset: the task for running experiments;
2. sync_interval: specifies the number of batches of local training to be performed between two sync rounds. If adaptive sync enabled, this is the frequency to be used at the start;
3. num_peers: specifies the number of peers required during each synchronization round for Swarm Learning to proceed further;
4. learning_rate: determines the step size at each iteration while moving toward a minimum of a loss function;
5. batch_size: determines the number of samples utilized in one iteration;
6. num_users: the num of peers in each experiment;

### 安装步骤
```bash
# 克隆仓库
git clone https://github.com/*repo/fedswp_tl.git
cd fedswp_tl

# 创建虚拟环境
python -m venv fedswp_tl
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 基础使用
```bash
# Running experiments
python main.py
python -m torch.distributed.launch --nnodes 2 --nproc_per_node 5 main.py
```
---

## 📜 Disclaimer (声明)
This project is based on the following open-source projects for secondary development, and we would like to express our gratitude to the related projects and research and development personnel.

**The resources related to this project are for academic research purposes only and are strictly prohibited for commercial use.** When using parts involving third-party code, please strictly follow the corresponding open-source agreements. The content generated by the model is affected by factors such as model calculation, randomness, and quantization accuracy loss. This project cannot guarantee its accuracy. For any content output by the model, this project does not assume any legal responsibility and does not assume responsibility for any losses that may result from the use of related resources and output results.

This project is initiated and maintained by individuals and collaborators in their spare time, so we cannot guarantee a timely response to resolving relevant issues.

## 🤝 Feedback (联系我们)

If you have any questions, please submit them in GitHub Issues.

- Before submitting a question, please check if the FAQ can solve the problem and consult past issues to see if they can help.
- Please use our dedicated issue template for submitting.
- Duplicate and unrelated issues will be handled by [stable-bot](https://github.com/marketplace/stale); please understand.
- Raise questions politely and help build a harmonious discussion community.
