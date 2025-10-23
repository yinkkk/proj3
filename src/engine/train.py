# ==============================================
# D:\Project\DeepLearning\proj3\src\engine\train.py
# ==============================================
import os,sys
import torch
import timm
import tqdm
import pathlib
from torch.utils.data import DataLoader

# ---------- 1. 以“本文件”为锚点，一次性算出根目录 ----------
FILE_ROOT = pathlib.Path(__file__).resolve()          # 当前文件
SRC_DIR   = FILE_ROOT.parent                          # ...\src\engine
PROJ_ROOT = SRC_DIR.parent.parent                     # ...\proj3

# ---------- 2. 所有路径全部用 Path，拼出来 ----------
CSV_TRAIN = PROJ_ROOT / 'data' / 'train.csv'
CSV_VAL   = PROJ_ROOT / 'data' / 'val.csv'
IMG_DIR   = PROJ_ROOT / 'data' / 'images'
CKPT_DIR  = PROJ_ROOT / 'experiments' / 'vit_baseline'

# 如果 experiments 下还有子目录，也一次性建好
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------- 3. 把项目根目录加入 PYTHONPATH，避免 import 报错 ----------
sys.path.insert(0, str(PROJ_ROOT))

# ---------- 4. 正常 import 你自己的模块 ----------
from src.data.dataset import FundusDataset
from src.models.vit import get_model
from src.losses import FocalLoss
from src.metrics import MetricTracker
from src.utils.seed import seed_everything

# ===============================
# 超参数
# ===============================
cfg = {
    # === data ===
    "img_size": 224,
    "batch_size": 32,
    "num_workers": 8,
    "n_classes": 20,

    # === model ===
    "arch": "vit_base_patch16_224",
    "pretrained": True,
    "drop_rate": 0.1,

    # === optimizer ===
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "epochs": 30,

    # === loss ===
    "loss_name": "focal",
    "focal_gamma": 2.0,
    "focal_alpha": None,

    # === misc ===
    "seed": 23,
    "ckpt_dir": CKPT_DIR,          # 直接用上面算好的路径
    "resume": None
}

# ===============================
# 训练主函数
# ===============================
def run(cfg):
    seed_everything(cfg['seed'])
    os.makedirs(cfg['ckpt_dir'], exist_ok=True)

    # ---- 数据 ----
    train_set = FundusDataset(CSV_TRAIN, IMG_DIR, img_size=224, mode='train')
    val_set   = FundusDataset(CSV_VAL,   IMG_DIR, img_size=224, mode='val')
    train_loader = DataLoader(train_set, cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=True)
    val_loader   = DataLoader(val_set,   cfg['batch_size'], shuffle=False,
                              num_workers=cfg['num_workers'], pin_memory=True)

    # ---- 模型 ----
    model = get_model(cfg['arch'], cfg['n_classes'],
                      pretrained=cfg['pretrained'],
                      drop_rate=cfg['drop_rate']).cuda()

    # ---- 损失 ----
    if cfg['loss_name'].lower() == 'focal':
        criterion = FocalLoss(alpha=None, gamma=cfg['focal_gamma']).cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()

    # ---- 优化器 & 调度器 ----
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'])

    # ---- 训练循环 ----
    best_auc = 0.
    for epoch in range(1, cfg['epochs'] + 1):
        # --- train ---
        model.train()
        for x, y in tqdm.tqdm(train_loader, desc=f'Epoch {epoch} [Train]'):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # --- val ---
        model.eval()
        metric = MetricTracker(cfg['n_classes'])
        with torch.no_grad():
            for x, y in tqdm.tqdm(val_loader, desc=f'Epoch {epoch} [Val]'):
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                logits = model(x)
                metric.update(logits, y)
        val_dict = metric.compute()
        print(f'Epoch {epoch:02d} | val_auc={val_dict["auc"]:.4f}')

        # --- save best ---
        if val_dict['auc'] > best_auc:
            best_auc = val_dict['auc']
            save_path = cfg['ckpt_dir'] / 'best_auc.pth'
            torch.save(model.state_dict(), save_path)
            print(f'✅ Saved new best model (AUC={best_auc:.4f}) -> {save_path}')

# ---------- 启动 ----------
if __name__ == '__main__':
    run(cfg)