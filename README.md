## 🚀 DKGTrack: Language Decoupling with Fine-grained Knowledge Guidance for Referring Multi-object Tracking

**DKGTrack** is a novel Referring Multi-Object Tracking (RMOT) framework that decouples language expressions into localized descriptions and motion states, enabling more precise object tracking guided by natural language.
<p align="center"><img src="./assets/framework.png" width="800"/></p>

## 🔧 Features

- **Static Semantic Enhancement (SSE):** Improves region-level vision-language alignment for more discriminative object representations.
- **Motion Perception Alignment (MPA):** Aligns motion expressions with object queries for temporally consistent tracking.

## 🛠️ Setup
git clone https://github.com/acyddl/DKGTrack.git

conda create -n DKGTrack python=3.8 -y

conda activate DKGTrack

For detailed installation instructions and dependencies, please refer to [install.md](https://github.com/acyddl/DKGTrack/blob/main/Install.md)

## 📅 Dataset
For instructions on downloading and organizing the dataset, please refer to the [TempRMOT repository](https://github.com/zyn213/TempRMOT).

### Training
Before training, you can download COCO pretrained weights from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

Then, train **DKGTrack** on Refer-KITTI using the following command:
```bash
sh configs/dkgtrack_rmot_train_rk.sh
```
Train **DKGTrack** on Refer-KITTI_v2 using the following command:
```bash
sh configs/dkgtrack_rmot_train.sh
```
### Inference
For evaluating DKGTrack on Refer-KITTI, run:
```bash
sh configs/dkgtrack_rmot_test_rk.sh
```
For evaluating DKGTrack on Refer-KITTI_v2, run:
```bash
sh configs/dkgtrack_rmot_test.sh
```
After testing, you can obtain the main results by running the evaluation scripts:
```bash
cd TrackEval/script
sh evaluate_rmot.sh
```
### Main Results

| **Method** | **Dataset** | **HOTA** | **DetA** | **AssA** | **DetRe** | **DetPr** | **AssRe** | **AssRe** | **LocA** |                                           **URL**                                           |
|:----------:|:-----------:|:--------:|:--------:|:--------:|:---------:|:---------:|:---------:|-----------|----------| :-----------------------------------------------------------------------------------------: |
| DKGTrack  | Refer-KITTI |  52.23   |  41.10   |  66.51   |   54.64   |   61.64  |   70.73   | 89.17     | 90.60   | [model](https://pan.baidu.com/s/1kKCDaUVa5BmsWWxpaR0j8w) (afd5) |
    
## 📜 License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) **Usage and License Notices**: The code are intended and licensed for research use only.

## Acknowledgement
We sincerely thank projects [RMOT](https://github.com/wudongming97/RMOT), [TempRMOT](https://github.com/zyn213/TempRMOT), [DsHmp](https://github.com/heshuting555/DsHmp) for providing their open-source resources.

## 📫 Contact

If you have any questions, feel free to open an issue or contact us.

