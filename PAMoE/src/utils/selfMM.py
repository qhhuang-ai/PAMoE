import torch

# 每个获取特征后食用
def update_labels(hp,labels=None,s=None,f=None,mode=None): # f为融合模态，s为单模态
    MIN = 1e-8

    def update_single_label(f_single, mode):
        d_sp = torch.norm(f_single - getattr(hp,mode + "_pos_center"), dim=-1)
        d_sn = torch.norm(f_single - getattr(hp,mode + "_neg_center"), dim=-1)
        delta_s = (d_sn - d_sp) / (d_sp + MIN)
        # d_s_pn = torch.norm(self.center_map[mode]['pos'] - self.center_map[mode]['neg'], dim=-1)
        # delta_s = (d_sn - d_sp) / (d_s_pn + MIN)
        alpha = delta_s / (delta_f + MIN)
        f_labels = hp.f_labels.cuda()
        new_labels = 0.5 * alpha * f_labels + \
                     0.5 * (f_labels + delta_s - delta_f)
        new_labels = torch.clamp(new_labels, min=-3.0, max=3.0)
        n = hp.nows
        labels = ((n - 1) / (n + 1) * getattr(hp,mode + "_labels").cuda() + 2 / (n + 1) * new_labels).clone().detach()
        setattr(hp,mode + "_labels",labels)


    keys = ["lv","la","al","av","va","vl","f"]
    if mode is None: #epoch为1，使用全局标签
        label = labels.clone().detach().view(-1)
        for k in keys:
            setattr(hp,k+"_labels",label)
    elif mode == "f": # 只更新融合标签
        label = labels.clone().detach().view(-1)
        for k in keys:
            setattr(hp, k + "_labels", label)
    else:
        with torch.no_grad():
            d_fp = torch.norm(f - hp.f_pos_center, dim=-1)
            d_fn = torch.norm(f - hp.f_neg_center, dim=-1)
            delta_f = (d_fn - d_fp) / (d_fp + MIN)
            update_single_label(s, mode=mode)

# 每个获取特征后食用,要在epoch 尾update_center后清为0
def update_features(hp,features,mode=None): #
    feature = features.clone().detach()
    label = getattr(hp,mode + "_labels")
    if len(label.shape) > 1:
        label = label.squeeze(dim=1)
    pos_idx = (label > 0)
    neg_idx = (label < 0)
    with torch.no_grad():
        if hasattr(hp,mode+"_pos_features"): # 有则加上
            if pos_idx.sum() > 0:
                pos_feature = getattr(hp,mode+"_pos_features") + feature[pos_idx].sum(dim=0)
                setattr(hp, mode + "_pos_features", pos_feature)
                setattr(hp, mode + "_pos_num",getattr(hp,mode + "_pos_num") +pos_idx.sum())
        else:
            if pos_idx.sum() > 0:   # 没有则设
                pos_feature = feature[pos_idx].sum(dim=0)
                setattr(hp, mode + "_pos_features", pos_feature)
                setattr(hp, mode + "_pos_num", pos_idx.sum())

        if hasattr(hp, mode + "_neg_features"):
            if neg_idx.sum() > 0:
                neg_feature = getattr(hp, mode + "_neg_features") + feature[neg_idx].sum(dim=0)
                setattr(hp, mode + "_neg_features", neg_feature)
                setattr(hp, mode + "_neg_num", getattr(hp, mode + "_neg_num") + neg_idx.sum())
        else:
            if neg_idx.sum() > 0:
                neg_feature = feature[neg_idx].sum(dim=0)
                setattr(hp, mode + "_neg_features", neg_feature)
                setattr(hp, mode + "_neg_num", neg_idx.sum())

# 每个epoch尾巴食用，并重置正负数量和特征为0和
def update_centers(hp,mode=None):
    if mode is None:
        keys = ['la', 'lv', 'al', 'av', 'va', 'vl', 'f']
    else:
        keys = [mode]
    with torch.no_grad():
        for k in keys:
            pos_cen = getattr(hp,k + "_pos_features")  / getattr(hp,k + "_pos_num")
            setattr(hp,k + "_pos_center",pos_cen)
            neg_cen = getattr(hp, k + "_neg_features") / getattr(hp, k + "_neg_num")
            setattr(hp, k + "_neg_center", neg_cen)
            setattr(hp,k + "_pos_features",0)
            setattr(hp, k + "_neg_features", 0)
            setattr(hp, k + "_pos_num", 0)
            setattr(hp, k + "_neg_num", 0)

#貌似不用，因为每次center会被重新赋值
def reset_centers(hp):
    keys = ['la', 'lv', 'al', 'av', 'va', 'vl', 'f']
    for k in keys:
        setattr(hp, k + "_pos_center", 0)
        setattr(hp, k + "_neg_center", 0)