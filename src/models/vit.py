import timm
import torch.nn as nn

def get_model(arch, n_classes, pretrained=True, drop_rate=0.1):
    model = timm.create_model(arch, pretrained=pretrained, drop_rate=drop_rate)
    # 把原 head 换成咱们类别数
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, n_classes)
    return model

if __name__ == '__main__':
    m = get_model('vit_base_patch16_224', 20)
    print(m.head)          # 快速自测