import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from pytorch_metric_learning.losses.ntxent_loss import NTXentLoss
from pytorch_metric_learning.utils import common_functions as c_f
import torch
import torch.nn as nn
import random

def get_all_pairs_indices(labels, ref_labels=None):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx


def convert_to_pairs(indices_tuple, labels):
    """
    This returns anchor-positive and anchor-negative indices,
    regardless of what the input indices_tuple is
    Args:
        indices_tuple: tuple of tensors. Each tensor is 1d and specifies indices
                        within a batch
        labels: a tensor which has the label for each element in a batch
    """
    if indices_tuple is None:
        return get_all_pairs_indices(labels)
    elif len(indices_tuple) == 4:
        return indices_tuple
    else:
        a, p, n = indices_tuple
        return a, p, a, n

# self.ntxent_loss = cont_NTXentLoss(temperature=self.heat)


class cont_NTXentLoss(NTXentLoss):
    def __init__(self, temperature=0.07,**kwargs):
        super().__init__()
        self.hp = kwargs['hp']
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)
        self.label = None

    def getIdx(self,labels): # 获取极性的索引值
        pos = []
        neg = []
        for i in range(labels.shape[0]):
            if labels[i] < 0:
                neg.append(i)
            elif labels[i] > 0:
                pos.append(i)
        dc = {'pos': pos, 'neg': neg}
        return dc
    def compute(self,common,t,a,v,labels):
        dc = self.getIdx(labels)
        with torch.no_grad():
            cosM = nn.CosineSimilarity(dim=-1,eps=1e-6)(torch.cat((t,a,v),dim=-1).unsqueeze(1)
                                                        ,torch.cat((t,a,v),dim=-1).unsqueeze(0))

        rankM = torch.zeros(cosM.shape)
        for i in range(len(cosM)):
            _, rankM[i, :] = torch.sort(cosM[i, :], descending=True)
        loss = 0
        left = 0
        ap = []
        an = []
        p1 = []
        n1 = []
        for i in range(len(t)):
            if labels[i] == 0:
                continue
            anchor = t[i]
            ref = dc['pos'] if labels[i] > 0 else dc['neg']
            posIdx = []
            negIdx = []
            for j in range(len(rankM[i,:])):
                if j == 0: #不包含自身
                    continue
                if rankM[i,j].item() in ref:
                    posIdx.append(rankM[i,j].item())
                else:
                    negIdx.append(rankM[i,j].item())
            if len(negIdx) == 0 or len(posIdx) == 0: #正负样本一定要有
                continue
            posIdx = posIdx[:5] if len(posIdx) >= 5 else posIdx
            negIdx = negIdx[:5] + list(reversed(negIdx))[:5] if len(negIdx) > 5 else negIdx

            negIdx = list(set(negIdx))  # 去重
            if self.hp.choose == 'random':
                pIdx = random.sample(posIdx, 3 if len(posIdx) >= 3 else len(posIdx))
                nIdx = random.sample(list(reversed(negIdx)), 2 if len(negIdx) >= 2 else len(negIdx)) + \
                   random.sample(negIdx, 2 if len(negIdx) >= 2 else len(negIdx))
            else:
                pIdx = posIdx[:3]
                nIdx = negIdx[:2] + list(reversed(negIdx))[:2]
            nIdx = list(set(nIdx))
            pIdx = [int(i) for i in pIdx]
            nIdx = [int(i) for i in nIdx]
            p = torch.cat((t[pIdx], v[pIdx], a[pIdx]), dim=0)
            n = torch.cat((t[nIdx], v[nIdx], a[nIdx], common[[i] + pIdx]), dim=0)
            if len(p) > 0:
                emb = torch.cat((anchor.unsqueeze(0), p, n), dim=0)
                left = left + 1
                ap = torch.tensor([0 for i in range(len(p))]).cuda()
                an = torch.tensor([0 for i in range(len(n))]).cuda()
                p1 = torch.tensor([i for i in range(len(p))]).cuda() + 1
                n1 = torch.tensor([i for i in range(len(n))]).cuda() +1 + len(p)
                indices_tuple = (ap,p1,an,n1)
                loss += self.compute_loss(emb,labels,indices_tuple,None,None)
        loss = loss / left
        return left
    def compute_loss(self, embeddings, labels, indices_tuple,ref_emb,ref_labels):
        # 7*6
        indices_tuple = convert_to_pairs(indices_tuple, labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings)
        return self.loss_method(mat, indices_tuple)['loss']['losses'].mean()

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, n = indices_tuple
        weight = 1
        if self.label is not None:
            weight = abs(self.label[a2] - self.label[n])

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            # if dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            # print(torch.sum(neg_pairs))
            if self.label is not None:
                neg_pairs = neg_pairs * weight / 2
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = c_f.to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = c_f.neg_inf(dtype)

            max_val = torch.max(
                pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
            ).detach()
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)

            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator / denominator) + c_f.small_val(dtype))
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()

# 模态内对比损失,从两种视角获取的同一模态表征，1：相同标签的尽可能近，不同标签的尽可能远，2：同一视角内的相同标签尽可能近，不同标签的尽可能远
class cont_Intraloss(NTXentLoss):
    def __init__(self, temperature=0.07,**kwargs):
        super().__init__()
        self.hp = kwargs['hp']
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)
        self.label = None

    def getIdx(self,labels): # 获取极性的索引值
        pos = []
        neg = []
        for i in range(labels.shape[0]):
            if labels[i] < 0:
                neg.append(i)
            elif labels[i] > 0:
                pos.append(i)
        dc = {'pos': pos, 'neg': neg}
        return dc
    def compute(self,a,b,labels):
        dc = self.getIdx(labels)
        with torch.no_grad():
            cosM = nn.CosineSimilarity(dim=-1,eps=1e-6)(a.unsqueeze(1)
                                                        ,b.unsqueeze(0))

        rankM = torch.zeros(cosM.shape)
        for i in range(len(cosM)):
            _, rankM[i, :] = torch.sort(cosM[i, :], descending=True)
        loss = 0
        left = 0
        ap = []
        an = []
        p1 = []
        n1 = []
        for i in range(len(a)):
            if labels[i] == 0:
                continue
            anchor = a[i]
            ref = dc['pos'] if labels[i] > 0 else dc['neg']
            posIdx = []
            negIdx = []
            for j in range(len(rankM[i,:])):
                # if j == 0: #与非自身计算时注释,不包含自身
                #     continue
                if rankM[i,j].item() in ref:
                    posIdx.append(rankM[i,j].item())
                else:
                    negIdx.append(rankM[i,j].item())
            posIdx = list(reversed(posIdx))[:5]
            if len(negIdx) == 0 or len(posIdx) == 0:
                continue
            negIdx = negIdx[:5] + list(reversed(negIdx))[:5] if len(negIdx) > 5 else negIdx
            negIdx = list(set(negIdx))  # 去重

            if self.hp.choose == 'random':
                if len(posIdx) == 0 or len(negIdx) == 0:
                    continue
                pIdx = random.sample(posIdx, 3 if len(posIdx) >= 3 else len(posIdx))
                nIdx = random.sample(negIdx, 3 if len(negIdx) >= 3 else len(negIdx)) + \
                   random.sample(negIdx[-6:-1], 1 if len(negIdx) >= 1 else len(negIdx))
            else:
                pIdx = posIdx[:3]
                nIdx = negIdx[:3] + [negIdx[-1]]
            nIdx = list(set(nIdx))
            pIdx = [int(i) for i in pIdx]
            nIdx = [int(i) for i in nIdx]
            p = b[pIdx]
            n = b[nIdx]
            if len(p) > 0 and len(n) > 0:
                emb = torch.cat((anchor.unsqueeze(0), p, n), dim=0)
                left = left + 1
                ap = torch.tensor([0 for i in range(len(p))]).cuda()
                an = torch.tensor([0 for i in range(len(n))]).cuda()
                p1 = torch.tensor([i for i in range(len(p))]).cuda() + 1
                n1 = torch.tensor([i for i in range(len(n))]).cuda() +1 + len(p)
                indices_tuple = (ap,p1,an,n1)
                loss += self.compute_loss(emb,labels,indices_tuple,None,None)
        if left == 0:
            loss = 0
        else:
            loss = loss / left
        return loss
    def compute_loss(self, embeddings, labels, indices_tuple,ref_emb,ref_labels):
        indices_tuple = convert_to_pairs(indices_tuple, labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings)
        return self.loss_method(mat, indices_tuple)['loss']['losses'].mean()

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, n = indices_tuple
        weight = 1
        if self.label is not None:
            weight = abs(self.label[a2] - self.label[n])

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            # if dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            # print(torch.sum(neg_pairs))
            if self.label is not None:
                neg_pairs = neg_pairs * weight / 2
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = c_f.to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = c_f.neg_inf(dtype)

            max_val = torch.max(
                pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
            ).detach()
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)

            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator / denominator) + c_f.small_val(dtype))
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()

class cont_Interloss(NTXentLoss):
    def __init__(self, temperature=0.07,**kwargs):
        super().__init__()
        self.hp = kwargs['hp']
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)
        self.label = None

    def getIdx(self,labels): # 获取极性的索引值
        pos = []
        neg = []
        for i in range(labels.shape[0]):
            if labels[i] < 0:
                neg.append(i)
            elif labels[i] > 0:
                pos.append(i)
        dc = {'pos': pos, 'neg': neg}
        return dc
    def compute(self,a,labels):
        dc = self.getIdx(labels)
        with torch.no_grad():
            cosM = nn.CosineSimilarity(dim=-1,eps=1e-6)(a.unsqueeze(1)
                                                        ,a.unsqueeze(0))

        rankM = torch.zeros(cosM.shape)
        for i in range(len(cosM)):
            _, rankM[i, :] = torch.sort(cosM[i, :], descending=True)
        loss = 0
        left = 0
        ap = []
        an = []
        p1 = []
        n1 = []
        for i in range(len(a)):
            if labels[i] == 0:
                continue
            anchor = a[i]
            ref = dc['pos'] if labels[i] > 0 else dc['neg']
            posIdx = []
            negIdx = []
            for j in range(len(rankM[i,:])):
                if j == 0: #不包含自身
                    continue
                if rankM[i,j].item() in ref:
                    posIdx.append(rankM[i,j].item())
                else:
                    negIdx.append(rankM[i,j].item())
            posIdx = list(reversed(posIdx))[:10]
            negIdx = negIdx[:5] + list(reversed(negIdx))[:5] if len(negIdx) > 5 else negIdx
            negIdx = list(set(negIdx))  # 去重
            if len(posIdx) <= 0 or len(negIdx) <= 0:
                continue
            if self.hp.choose == 'random':
                pIdx = random.sample(posIdx, 3 if len(posIdx) >= 3 else len(posIdx))
                nIdx = random.sample(negIdx, 3 if len(negIdx) >= 3 else len(negIdx)) + \
                   random.sample(negIdx[-6:-1], 1 if len(negIdx) >= 1 else len(negIdx))
            else:
                pIdx = posIdx[:3]
                nIdx = negIdx[:3] + [negIdx[-1]]
            nIdx = list(set(nIdx))
            pIdx = [int(i) for i in pIdx]
            nIdx = [int(i) for i in nIdx]
            p = a[pIdx]
            n = a[nIdx]
            if len(p) > 0 and len(n) > 0:
                emb = torch.cat((anchor.unsqueeze(0), p, n), dim=0)
                left = left + 1
                ap = torch.tensor([0 for i in range(len(p))]).cuda()
                an = torch.tensor([0 for i in range(len(n))]).cuda()
                p1 = torch.tensor([i for i in range(len(p))]).cuda() + 1
                n1 = torch.tensor([i for i in range(len(n))]).cuda() +1 + len(p)
                indices_tuple = (ap,p1,an,n1)
                loss += self.compute_loss(emb,labels,indices_tuple,None,None)
        if left == 0:
            loss = 0
        else:
            loss = loss / left
        return loss

    def compute_loss(self, embeddings, labels, indices_tuple,ref_emb,ref_labels):
        indices_tuple = convert_to_pairs(indices_tuple, labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings)
        return self.loss_method(mat, indices_tuple)['loss']['losses'].mean()

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, n = indices_tuple
        weight = 1
        if self.label is not None:
            weight = abs(self.label[a2] - self.label[n])

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            # if dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            # print(torch.sum(neg_pairs))
            if self.label is not None:
                neg_pairs = neg_pairs * weight / 2
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = c_f.to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = c_f.neg_inf(dtype)

            max_val = torch.max(
                pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
            ).detach()
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)

            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator / denominator) + c_f.small_val(dtype))
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()

# la与al和av跟lv与vl以及va
class cont_Jointloss(NTXentLoss):
    def __init__(self, temperature=0.07,**kwargs):
        super().__init__()
        self.hp = kwargs['hp']
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)
        self.label = None

    def getIdx(self,labels): # 获取极性的索引值
        pos = []
        neg = []
        for i in range(labels.shape[0]):
            if labels[i] < 0:
                neg.append(i)
            elif labels[i] > 0:
                pos.append(i)
        dc = {'pos': pos, 'neg': neg}
        return dc
    def compute(self,ta,at,av,tv,vt,va,labels):     #ta和at以及av对比，tv和va以及va对比
        dc = self.getIdx(labels)
        with torch.no_grad():
            taat_cosM = nn.CosineSimilarity(dim=-1,eps=1e-6)(ta.unsqueeze(1)
                                                        ,at.unsqueeze(0))
            taav_cosM = nn.CosineSimilarity(dim=-1, eps=1e-6)(ta.unsqueeze(1)
                                                              , av.unsqueeze(0))
            tvvt_cosM = nn.CosineSimilarity(dim=-1, eps=1e-6)(tv.unsqueeze(1)
                                                              , vt.unsqueeze(0))
            tvva_cosM = nn.CosineSimilarity(dim=-1, eps=1e-6)(tv.unsqueeze(1)
                                                              , va.unsqueeze(0))

        taat_rankM = torch.zeros(taat_cosM.shape)
        taav_rankM = torch.zeros(taav_cosM.shape)
        tvvt_rankM = torch.zeros(tvvt_cosM.shape)
        tvva_rankM = torch.zeros(tvva_cosM.shape)
        for i in range(len(taat_cosM)):
            _, taat_rankM[i, :] = torch.sort(taat_cosM[i, :], descending=True)
        for i in range(len(taav_cosM)):
            _, taav_rankM[i, :] = torch.sort(taav_cosM[i, :], descending=True)
        for i in range(len(tvvt_cosM)):
            _, tvvt_rankM[i, :] = torch.sort(tvvt_cosM[i, :], descending=True)
        for i in range(len(tvva_cosM)):
            _, tvva_rankM[i, :] = torch.sort(tvva_cosM[i, :], descending=True)

        taloss = 0
        left = 0
        ap = []
        an = []
        p1 = []
        n1 = []
        for i in range(len(ta)):
            if labels[i] == 0:
                continue
            anchor = ta[i]
            ref = dc['pos'] if labels[i] > 0 else dc['neg']
            atposIdx = []
            atnegIdx = []
            for j in range(len(taat_rankM[i,:])):
                if j == 0: #不包含自身
                    continue
                if taat_rankM[i,j].item() in ref:
                    atposIdx.append(taat_rankM[i,j].item())
                else:
                    atnegIdx.append(taat_rankM[i,j].item())
            atposIdx = list(reversed(atposIdx))[:10]
            atnegIdx = atnegIdx[:5] + list(reversed(atnegIdx))[:5] if len(atnegIdx) > 5 else atnegIdx
            atnegIdx = list(set(atnegIdx))  # 去重

            avposIdx = []
            avnegIdx = []
            for j in range(len(taat_rankM[i, :])):
                if j == 0:  # 不包含自身
                    continue
                if taav_rankM[i, j].item() in ref:
                    avposIdx.append(taav_rankM[i, j].item())
                else:
                    avnegIdx.append(taav_rankM[i, j].item())
            avposIdx = list(reversed(atposIdx))[:10]
            avnegIdx = atnegIdx[:5] + list(reversed(atnegIdx))[:5] if len(atnegIdx) > 5 else atnegIdx
            avnegIdx = list(set(avnegIdx))  # 去重
            #分子分母都要大于0
            if min(len(atposIdx) ,len(avposIdx)) <= 0 or min(len(avnegIdx),len(atnegIdx)) <= 0:
                continue

            atpIdx = atposIdx[:3]
            atnIdx = list(set(atnegIdx[:3] + [atnegIdx[-1]])) if len(atnegIdx) > 0 else []
            atpIdx = [int(i) for i in atpIdx]
            atnIdx = [int(i) for i in atnIdx]
            atp = at[atpIdx]
            atn = at[atnIdx]

            avpIdx = avposIdx[:3]
            avnIdx = list(set(avnegIdx[:3] + [avnegIdx[-1]]))  if len(avnegIdx) > 0 else []
            avpIdx = [int(i) for i in avpIdx]
            avnIdx = [int(i) for i in avnIdx]
            avp = av[avpIdx]
            avn = av[avnIdx]
            # 加上自身样本两个模态作正样本
            p = torch.cat((atp,avp,at[i].unsqueeze(0),av[i].unsqueeze(0)),dim = 0)
            n = torch.cat((atn,avn),dim = 0)
            if len(p) > 0 and len(n) > 0:
                emb = torch.cat((anchor.unsqueeze(0), p, n), dim=0)
                left = left + 1
                ap = torch.tensor([0 for i in range(len(p))]).cuda()
                an = torch.tensor([0 for i in range(len(n))]).cuda()
                p1 = torch.tensor([i for i in range(len(p))]).cuda() + 1
                n1 = torch.tensor([i for i in range(len(n))]).cuda() +1 + len(p)
                indices_tuple = (ap,p1,an,n1)
                taloss += self.compute_loss(emb,labels,indices_tuple,None,None)
        if left == 0:
            taloss = 0
        else:
            taloss = taloss / left
        taloss = taloss*self.hp.ta

        tvloss = 0
        left = 0
        ap = []
        an = []
        p1 = []
        n1 = []
        for i in range(len(ta)):
            if labels[i] == 0:
                continue
            anchor = ta[i]
            ref = dc['pos'] if labels[i] > 0 else dc['neg']
            vaposIdx = []
            vanegIdx = []
            for j in range(len(tvva_rankM[i, :])):
                if j == 0:  # 不包含自身
                    continue
                if tvva_rankM[i, j].item() in ref:
                    vaposIdx.append(tvva_rankM[i, j].item())
                else:
                    vanegIdx.append(tvva_rankM[i, j].item())
            vaposIdx = list(reversed(vaposIdx))[:10]
            vanegIdx = vanegIdx[:5] + list(reversed(vanegIdx))[:5] if len(vanegIdx) > 5 else vanegIdx
            vanegIdx = list(set(vanegIdx))  # 去重

            vtposIdx = []
            vtnegIdx = []
            for j in range(len(tvvt_rankM[i, :])):
                if j == 0:  # 不包含自身
                    continue
                if taav_rankM[i, j].item() in ref:
                    vtposIdx.append(tvvt_rankM[i, j].item())
                else:
                    vtnegIdx.append(tvvt_rankM[i, j].item())
            vtposIdx = list(reversed(vtposIdx))[:10]
            vtnegIdx = vtnegIdx[:5] + list(reversed(vtnegIdx))[:5] if len(vtnegIdx) > 5 else vtnegIdx
            vtnegIdx = list(set(vtnegIdx))  # 去重
            # 分子分母都要大于0
            if min(len(vtposIdx), len(vtposIdx)) <= 0 or min(len(vtnegIdx), len(vtnegIdx)) <= 0:
                continue

            vapIdx = vaposIdx[:3]
            vanIdx = list(set(vanegIdx[:3] + [vanegIdx[-1]])) if len(vanegIdx) > 0 else []
            vapIdx = [int(i) for i in vapIdx]
            vanIdx = [int(i) for i in vanIdx]
            vap = va[vapIdx]
            van = va[vanIdx]

            vtpIdx = vtposIdx[:3]
            vtnIdx = list(set(vtnegIdx[:3] + [vtnegIdx[-1]])) if len(vtnegIdx) > 0 else []
            vtpIdx = [int(i) for i in vtpIdx]
            vtnIdx = [int(i) for i in vtnIdx]
            vtp = vt[vtpIdx]
            vtn = vt[vtnIdx]
            # 加上自身样本两个模态作正样本
            p = torch.cat((vtp, vap, va[i].unsqueeze(0), vt[i].unsqueeze(0)), dim=0)
            n = torch.cat((van, vtn), dim=0)
            if len(p) > 0 and len(n) > 0:
                emb = torch.cat((anchor.unsqueeze(0), p, n), dim=0)
                left = left + 1
                ap = torch.tensor([0 for i in range(len(p))]).cuda()
                an = torch.tensor([0 for i in range(len(n))]).cuda()
                p1 = torch.tensor([i for i in range(len(p))]).cuda() + 1
                n1 = torch.tensor([i for i in range(len(n))]).cuda() + 1 + len(p)
                indices_tuple = (ap, p1, an, n1)
                tvloss += self.compute_loss(emb, labels, indices_tuple, None, None)
        if left == 0:
            tvloss = 0
        else:
            tvloss = tvloss / left
        tvloss = tvloss * self.hp.tv
        loss = tvloss + taloss
        return loss

    def compute_loss(self, embeddings, labels, indices_tuple,ref_emb,ref_labels):
        indices_tuple = convert_to_pairs(indices_tuple, labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings)
        return self.loss_method(mat, indices_tuple)['loss']['losses'].mean()

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, n = indices_tuple
        weight = 1
        if self.label is not None:
            weight = abs(self.label[a2] - self.label[n])

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            # if dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            # print(torch.sum(neg_pairs))
            if self.label is not None:
                neg_pairs = neg_pairs * weight / 2
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = c_f.to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = c_f.neg_inf(dtype)

            max_val = torch.max(
                pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
            ).detach()
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)

            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator / denominator) + c_f.small_val(dtype))
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()