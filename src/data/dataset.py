import pandas as pd, cv2
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

class FundusDataset(Dataset):
    def __init__(self, csv_file, img_dir, img_size=224, mode='train'):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.mode = mode

        self.label_cols = self.df.columns[1:]  # 第一列是 'path'，其余列是标签

        if mode == 'train':
            self.aug = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 使用 'path' 列直接作为图像路径
        img_path = self.img_dir / row['path']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode in ['train', 'val']:
            # 多分类/多标签情况
            label = torch.tensor(row[self.label_cols].values, dtype=torch.float32)
            return self.aug(image=image)['image'], label
        else:  # 测试集无标签
            return self.aug(image=image)['image'], row['path']
