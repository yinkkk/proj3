import torch, sklearn.metrics as skm

class MetricTracker:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        self.prob, self.tgt = [], []

    def update(self, logits, y):
        self.prob.append(torch.softmax(logits, dim=1).detach().cpu())
        self.tgt.append(y.detach().cpu())

    def compute(self):
        prob = torch.cat(self.prob).numpy()
        tgt  = torch.cat(self.tgt).numpy()
        auc = skm.roc_auc_score(tgt, prob, multi_class='ovr', average='macro')
        pred = prob.argmax(1)
        acc = skm.accuracy_score(tgt, pred)
        f1  = skm.f1_score(tgt, pred, average='macro')
        return dict(auc=auc, acc=acc, f1=f1)