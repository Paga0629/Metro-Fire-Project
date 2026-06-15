# Metro-Fire-Project
Physical AI &amp; Hologram-based Smart Fire Evacuation System for Unmanned Subways

## ⚙️ Setup
### 1. Dependencies and Installation
We recommend using `Python>=3.10`
```bash
conda create --name metro_fire python=3.10
conda activate metro_fire
pip install -U pip

# Install requirements in each folders (AECR-Net, haze-synthesis, YOLO)
pip install -r requirements.txt
```
Please check for details 
- [AECR-Net](https://github.com/GlassyWu/AECR-Net.git)
- [haze-synthesis](https://github.com/tranleanh/haze-synthesis.git)
- [yolov12](https://github.com/sunsmarterjie/yolov12.git)

## Datasets
We used MetroStation dataset for training and inference.
![MetroStation Dataset](assets/MetroStation.png)
You can download at the link below.
- [MetroStation](https://figshare.com/articles/dataset/MetroStation_Dataset/20521848?file=36732258)

## 🤗 Acknowledgements
We thank the authors of the following projects!
- [AECR-Net](https://github.com/GlassyWu/AECR-Net.git)
- [haze-synthesis](https://github.com/tranleanh/haze-synthesis.git)
- [yolov12](https://github.com/sunsmarterjie/yolov12.git)


![Results](assets/Results.png)
