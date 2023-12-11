import torch
import torch.nn.functional as F
import time

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
from .transformer import TransformerEncoder
from torch.distributions.normal import Normal
from utils.Nxloss import cont_Intraloss,cont_Interloss,cont_Jointloss
import numpy as np
from utils.selfMM import update_features,update_centers,update_labels
class SparseDispatcher(object):


    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):


        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "glove" or "Bert"
    """

    def __init__(self, hp):
        super(LanguageEmbeddingLayer, self).__init__()

        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        if hp.load == 1:
            self.load()
    def forward(self, sentences, bert_sent, bert_sent_type, bert_sent_mask):
        bert_output = self.bertmodel(input_ids=bert_sent,
                                     attention_mask=bert_sent_mask,
                                     token_type_ids=bert_sent_type)
        bert_output = bert_output[0]
        return bert_output  # return head (sequence representation)

    def load(self):
        a = torch.load("/home/lzh/datasets/MSA/MSA-BERT/epoch100_emb.pt")
        a["position_ids"] = self.bertmodel.embeddings.position_ids
        self.bertmodel.embeddings.load_state_dict(a)
        self.bertmodel.encoder.load_state_dict(torch.load("/home/lzh/datasets/MSA/MSA-BERT/epoch100_encoder.pt"))
class SubNet(nn.Module):  # 融合模块  可以更换！！！
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, n_class, dropout, modal_name='text'):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        # self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # normed = self.norm(x)
        dropped = self.drop(x)
        y_1 = torch.tanh(self.linear_1(dropped))
        fusion = self.linear_2(y_1)
        y_2 = torch.tanh(self.linear_2(y_1))
        y_3 = self.linear_3(y_2)
        return y_2, y_3


class FusionTrans(nn.Module):
    def __init__(self, hp, n_class):
        super(FusionTrans, self).__init__()
        self.hp = hp
        self.joint_loss = cont_Jointloss(temperature=self.hp.heat,hp=hp)
        self.ntxent_loss3 = cont_Interloss(temperature=self.hp.heat, hp=hp)
        self.ntxent_loss2 = cont_Interloss(temperature=self.hp.heat, hp=hp)
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.vonly = hp.vonly
        self.aonly = hp.aonly
        self.lonly = hp.lonly
        # 5
        self.num_heads = hp.num_heads
        # 5
        self.layers = hp.layers
        # 0.1
        self.attn_dropout = hp.attn_dropout
        # 0
        self.attn_dropout_a = hp.attn_dropout_a
        # 0
        self.attn_dropout_v = hp.attn_dropout_v
        # 0.1
        self.relu_dropout = hp.relu_dropout
        # 0.1
        self.res_dropout = hp.res_dropout
        # 0
        self.out_dropout = hp.out_dropout
        # 128
        self.d_prjh = hp.d_prjh
        # 0.25
        self.embed_dropout = hp.embed_dropout
        self.attn_mask = hp.attn_mask
        # 0
        self.n_lv = hp.n_tv
        # 1
        self.n_la = hp.n_ta

        combined_dim = 2 * self.d_l  # assuming d_l == d_a == d_v == 30

        # 1. Temporal convolutional layers
        # 768,
        self.proj_l = nn.Conv1d(hp.d_tin, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(hp.d_ain, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(hp.d_vin, self.d_v, kernel_size=1, padding=0, bias=False)

        #  Crossmodal Transformer（Attentions在里面）  CM
        # self.trans_l_with_a = self.get_network(self_type='la')
        # self.trans_l_with_v = self.get_network(self_type='lv')
        # self.trans_l_with_al = self.get_network(self_type='lla')
        # self.trans_l_with_vl = self.get_network(self_type='llv')

        self.t2a = self.get_network(self_type='la',layers=self.hp.layer_l)
        self.t2v = self.get_network(self_type='lv',layers=self.hp.layer_l)
        self.v2a = self.get_network(self_type='va',layers=self.hp.layer_v)
        self.v2t = self.get_network(self_type='vl',layers=self.hp.layer_v)
        self.a2v = self.get_network(self_type='av',layers=self.hp.layer_a)
        self.a2l = self.get_network(self_type='al',layers=self.hp.layer_a)

        # 首次融合后的映射，用于对比学习原为pro2
        if self.hp.expert == 'CONSMLP' : #6个MLP，每个MLP有n个专家
            if self.hp.polarity >= 0:  # 不同类别共用优化器,只区分极性
                self.classification = nn.ModuleList([nn.Sequential(
                    nn.Linear(30, 60),
                    nn.ReLU(),
                    nn.Linear(60, 1)
                ) for i in range(6)])
                # # 取中间ReLU层输出
                # self.center = nn.ModuleList([nn.Sequential(*list(self.classification[i]())[:2]) for i in range(6)])
            self.lv_router = nn.Linear(30, hp.expert_num, bias=False)
            self.vl_router = nn.Linear(30, hp.expert_num, bias=False)
            self.va_router = nn.Linear(30, hp.expert_num, bias=False)
            self.la_router = nn.Linear(30, hp.expert_num, bias=False)
            self.al_router = nn.Linear(30, hp.expert_num, bias=False)
            self.av_router = nn.Linear(30, hp.expert_num, bias=False)
            if self.hp.noise_aux > 0:
                self.lvnoise = nn.Linear(30, hp.expert_num, bias=False)
                self.avnoise = nn.Linear(30, hp.expert_num, bias=False)
                self.lanoise = nn.Linear(30, hp.expert_num, bias=False)
                self.alnoise = nn.Linear(30, hp.expert_num, bias=False)
                self.vanoise = nn.Linear(30, hp.expert_num, bias=False)
                self.vlnoise = nn.Linear(30, hp.expert_num, bias=False)
            self.softplus = nn.Softplus()
            self.softmax = nn.Softmax(1)
            self.register_buffer("mean", torch.tensor([0.0]))
            self.register_buffer("std", torch.tensor([1.0]))
            if self.hp.conln: # 对比映射是否使用归一化层
                self.con_pro = nn.ModuleList([nn.ModuleList([nn.Sequential(
                    nn.LayerNorm(30),
                    nn.Linear(30, 30),
                    nn.ReLU(),
                    nn.Dropout(p=self.hp.condrop)
                ) for j in range(self.hp.expert_num)]) for i in range(6)])
            else:
                self.con_pro = nn.ModuleList([nn.ModuleList([nn.Sequential(
                    nn.Linear(30, 30),
                    nn.ReLU(),
                    nn.Dropout(p=self.hp.condrop)
                ) for j in range(self.hp.expert_num)]) for i in range(6)])

        else:
            self.con_pro = nn.ModuleList([nn.Sequential(
                nn.LayerNorm(30),
                nn.Linear(30,30),
                nn.ReLU(),
                nn.Dropout(p=self.hp.condrop)
            ) for i in range(6)]) if self.hp.conln else nn.ModuleList([nn.Sequential(
                nn.Linear(30,30),
                nn.ReLU(),
                nn.Dropout(p=self.hp.condrop)
            ) for i in range(6)])



        if self.hp.interaction == 'add':
            pron_dim = 30
        else:
            pron_dim = 60
        self.prona1 = nn.Linear(pron_dim, 60)
        self.pronb1 = nn.Linear(pron_dim, 60)
        self.pronc1 = nn.Linear(pron_dim, 60)
        # add:30->60->120,cat:

        self.pronabc = nn.Linear(60, 120) if self.hp.fin_interaction == 'add' else nn.Sequential(nn.Linear(3*pron_dim,60),
                                                                                                 nn.ReLU(),
                                                                                                 nn.Linear(60,120))

        # Projection layers
        # self.proj1 = nn.Linear(self.d_l, self.d_l)  # 30,30
        #  再加一层映射！！！  180到128的
        d = 120
        self.proj3 = nn.Linear(d ,self.d_prjh)  # 128
        if hp.interaction == 'cat': # 中级融合 ，v = cat(vl,va)
            cat_dim =  60
        else: # add v = vl+va
            cat_dim = 30
        if self.hp.out == 'cat': # out = cat(v,l,a)
            cat_dim =  cat_dim*3
        else: #add out = v+l+a
            cat_dim = cat_dim
        self.out = nn.Sequential(
            nn.Linear(cat_dim, 80),
            nn.ReLU(),
            nn.Linear(80, 1)
        )


    def get_network(self, self_type='l', layers=-1,embed_dims = None):
        # self.d_l, self.d_a, self.d_v = hyp_params.embed_dim
        if self_type in ['la','lv']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['al', 'av']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['vl','va']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        if layers > -1 :
            return TransformerEncoder(embed_dim=embed_dim,  # 30
                                      num_heads=self.num_heads,
                                      layers=layers,
                                      attn_dropout=attn_dropout,
                                      relu_dropout=self.relu_dropout,
                                      res_dropout=self.res_dropout,
                                      embed_dropout=self.embed_dropout,
                                      attn_mask=self.attn_mask)

        return TransformerEncoder(embed_dim=embed_dim,  # 30
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.hp.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2,mode='lv'):
        # 均值or方差的计算
        route = getattr(self,mode+'_router')
        if self.hp.noise_aux > 0:
            noise = getattr(self,mode+'noise')
        clean_logits = route(x)
        # 一个batch中每个专家出现概率和
        mE = clean_logits.mean(dim=0)
        if self.hp.noise_aux > 0 and self.training:
            raw_noise_stddev = noise(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.hp.k + 1, self.hp.expert_num), dim=1)
        top_k_logits = top_logits[:, :self.hp.k]
        top_k_indices = top_indices[:, :self.hp.k]
        top_k_gates = self.softmax(top_k_logits)

        # 把非tok的置为0
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.hp.noise_aux > 0 and self.hp.k < self.hp.expert_num and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else :
            load = self._gates_to_load(gates)
        return gates, load,mE
    def weighted_loss(self, y_pred,y_true,mode ,idx1,idx2 = None):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1).cuda()
        a = getattr(self.hp,mode + "_labels")[idx1]
        b = getattr(self.hp,"f" + "_labels")[idx1]
        if idx2 is not None:
            a = a[idx2]
            b = b[idx2]
        a = a.cuda()
        b = b.cuda()
        weighted = torch.tanh(torch.abs(a - b))
        loss = torch.mean(weighted * torch.abs(y_pred - y_true))
        return loss
    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)
    def forward(self, l, a, v,labels):
        """
        传入格式：(seq_len, batch_size,emb_size)
        t: torch.Size([50, 32, 768])
        a: torch.Size([134, 32, 5])
        v: torch.Size([161, 32, 20])
        """
        if self.hp.polarity >= 0 and self.hp.expert == 'CONSMLP' and self.training and self.hp.stage != 0:
            update_labels(self.hp, labels) #先统一设为多模态标签




        text = self.proj_l(l.permute(1, 2, 0))  # torch.Size([32, 30, 50])   传入、得到 batch_size, n_feature,seq_len
        acoustic = self.proj_a(a.permute(1, 2, 0))  # torch.Size([32, 30, 147])
        visual = self.proj_v(v.permute(1, 2, 0))  # torch.Size([32, 30, 176])

        text = text.permute(2, 0, 1)  # seq_len,batch-size,n_feature
        acoustic = acoustic.permute(2, 0, 1)
        visual = visual.permute(2, 0, 1)

        #  Crossmodal Transformer
        # (V,A) --> L
        # l_with_a = self.trans_l_with_a(text, acoustic, acoustic)  # 50,32,30 Dimension (L, N, d_l)
        # l_with_v = self.trans_l_with_v(text, visual, visual)  # 50,32,30 Dimension (L, N, d_l)
        # 初级融合
        la = self.t2a(text, acoustic, acoustic)
        lv = self.t2a(text, visual, visual)

        al = self.a2l(acoustic, text, text)
        av = self.a2l(acoustic, visual, visual)

        vl = self.a2l(visual, text, text)
        va = self.a2l(visual, acoustic, acoustic) #L,N,30
        # if self.time == 'mean':
        #     v = (vl + va).mean(dim=0)
        #     l = (la + lv).mean(dim=0)
        #     a = (al + av).mean(dim=0)
        # else:
        l_con = 0
        sparse_aux_loss = 0
        aux_loss = 0
        l_class = 0
        _,batch,_ =  la.shape
        la_c,lv_c,al_c,av_c,vl_c,va_c = 0,0,0,0,0,0

        la_c = la.mean(dim=0) if self.hp.mode == 'mean' else la[-1]
        al_c = al.mean(dim=0) if self.hp.mode == 'mean' else al[-1]
        av_c = av.mean(dim=0) if self.hp.mode == 'mean' else av[-1]
        lv_c = lv.mean(dim=0) if self.hp.mode == 'mean' else lv[-1]
        vl_c = vl.mean(dim=0) if self.hp.mode == 'mean' else vl[-1]
        va_c = va.mean(dim=0) if self.hp.mode == 'mean' else va[-1]
        if self.hp.stage != 0 :
            if self.hp.expert != 'CONSMLP':
                la_c = self.con_pro[0](la_c)
                al_c = self.con_pro[1](al_c)
                av_c = self.con_pro[2](av_c)
                lv_c = self.con_pro[3](lv_c)
                vl_c = self.con_pro[4](vl_c)
                va_c = self.con_pro[5](va_c)
            else:
                lagates, laload, lamE = self.noisy_top_k_gating(la_c, self.training, mode='la')
                laimportance = lagates.sum(0)
                # 找出选中某个专家并且在选中后的排序中中为pos或neg的样本索引：找出所有选中该专家的样本索引，在获取该索引样本的标签获取该索引下积极样本索引

                algates, alload, almE = self.noisy_top_k_gating(al_c, self.training, mode='al')
                alimportance = algates.sum(0)

                avgates, avload, avmE = self.noisy_top_k_gating(av_c, self.training, mode='av')
                avimportance = avgates.sum(0)

                lvgates, lvload, lvmE = self.noisy_top_k_gating(lv_c, self.training, mode='lv')
                lvimportance = lvgates.sum(0)

                vlgates, vlload, vlmE = self.noisy_top_k_gating(vl_c, self.training, mode='vl')
                vlimportance = vlgates.sum(0)

                vagates, vaload, vamE = self.noisy_top_k_gating(va_c, self.training, mode='va')
                vaimportance = vagates.sum(0)
                if self.hp.fixed == 1:
                    ks = ["la","al","va","av","lv","vl"]
                    for k in ks:
                        x = locals()[k+'gates']
                        self.fix(x)
                if self.hp.expert_polarity == 'p' and self.hp.stage != 0:
                    lapos_expert = lagates[:, 0] > 0
                    laneg_expert = lagates[:, 1] > 0
                    laall_expert = lagates[:, 2] > 0
                    lapos_idx = (labels[lapos_expert] >= 0).squeeze()
                    laneg_idx = (labels[laneg_expert] < 0).squeeze()
                    laall_idx = laall_expert

                    lvpos_expert = lvgates[:, 0] > 0
                    lvneg_expert = lvgates[:, 1] > 0
                    lvall_expert = lvgates[:, 2] > 0
                    lvpos_idx = (labels[lvpos_expert] >= 0).squeeze()
                    lvneg_idx = (labels[lvneg_expert] < 0).squeeze()
                    lvall_idx = lvall_expert

                    alpos_expert = algates[:, 0] > 0
                    alneg_expert = algates[:, 1] > 0
                    alall_expert = algates[:, 2] > 0
                    alpos_idx = (labels[alpos_expert] >= 0).squeeze()
                    alneg_idx = (labels[alneg_expert] < 0).squeeze()
                    alall_idx = alall_expert

                    avpos_expert = avgates[:, 0] > 0
                    avneg_expert = avgates[:, 1] > 0
                    avall_expert = avgates[:, 2] > 0
                    avpos_idx = (labels[avpos_expert] >= 0).squeeze()
                    avneg_idx = (labels[avneg_expert] < 0).squeeze()
                    avall_idx = avall_expert

                    vapos_expert = vagates[:, 0] > 0
                    vaneg_expert = vagates[:, 1] > 0
                    vaall_expert = vagates[:, 2] > 0
                    vapos_idx = (labels[vapos_expert] >= 0).squeeze()
                    vaneg_idx = (labels[vaneg_expert] < 0).squeeze()
                    vaall_idx = vaall_expert

                    vlpos_expert = vlgates[:, 0] > 0
                    vlneg_expert = vlgates[:, 1] > 0
                    vlpos_idx = (labels[vlpos_expert] >= 0).squeeze()
                    vlneg_idx = (labels[vlneg_expert] < 0).squeeze()
                    vlall_idx = vlneg_expert
                elif self.hp.expert_polarity == 'n' and self.hp.stage != 0:
                    lapos_expert = lagates[:, 0] > 0
                    laneg_expert = lagates[:, 1] > 0
                    laall_expert = lagates[:, 2] > 0
                    lapos_idx = (labels[lapos_expert] > 0).squeeze()
                    laneg_idx = (labels[laneg_expert] <= 0).squeeze()
                    laall_idx = laall_expert

                    lvpos_expert = lvgates[:, 0] > 0
                    lvneg_expert = lvgates[:, 1] > 0
                    lvall_expert = lvgates[:, 2] > 0
                    lvpos_idx = (labels[lvpos_expert] > 0).squeeze()
                    lvneg_idx = (labels[lvneg_expert] <= 0).squeeze()
                    lvall_idx = lvall_expert

                    alpos_expert = algates[:, 0] > 0
                    alneg_expert = algates[:, 1] > 0
                    alall_expert = algates[:, 2] > 0
                    alpos_idx = (labels[alpos_expert] > 0).squeeze()
                    alneg_idx = (labels[alneg_expert] <= 0).squeeze()
                    alall_idx = alall_expert

                    avpos_expert = avgates[:, 0] > 0
                    avneg_expert = avgates[:, 1] > 0
                    avall_expert = avgates[:, 2] > 0
                    avpos_idx = (labels[avpos_expert] > 0).squeeze()
                    avneg_idx = (labels[avneg_expert] <= 0).squeeze()
                    avall_idx = avall_expert

                    vapos_expert = vagates[:, 0] > 0
                    vaneg_expert = vagates[:, 1] > 0
                    vaall_expert = vagates[:, 2] > 0
                    vapos_idx = (labels[vapos_expert] > 0).squeeze()
                    vaneg_idx = (labels[vaneg_expert] <= 0).squeeze()
                    vaall_idx = vaall_expert

                    vlpos_expert = vlgates[:, 0] > 0
                    vlneg_expert = vlgates[:, 1] > 0
                    vlall_expert = vlgates[:, 2] > 0
                    vlpos_idx = (labels[vlpos_expert] > 0).squeeze()
                    vlneg_idx = (labels[vlneg_expert] <= 0).squeeze()
                    vlall_idx = vlall_expert
                elif self.hp.stage != 0 :
                    lapos_expert = lagates[:, 0] > 0
                    laneg_expert = lagates[:, 1] > 0
                    laall_expert = lagates[:, 2] > 0
                    lapos_idx = (labels[lapos_expert] > 0).squeeze()
                    laneg_idx = (labels[laneg_expert] < 0).squeeze()
                    # laall_idx = laall_expert
                    lanot0_idx = (labels[laall_expert] != 0).squeeze()
                    lais0_idx = (labels[laall_expert] == 0).squeeze()

                    lvpos_expert = lvgates[:, 0] > 0
                    lvneg_expert = lvgates[:, 1] > 0
                    lvall_expert = lvgates[:, 2] > 0
                    lvpos_idx = (labels[lvpos_expert] > 0).squeeze()
                    lvneg_idx = (labels[lvneg_expert] < 0).squeeze()
                    lvnot0_idx = (labels[lvall_expert] != 0).squeeze()
                    lvis0_idx = (labels[lvall_expert] == 0).squeeze()

                    alpos_expert = algates[:, 0] > 0
                    alneg_expert = algates[:, 1] > 0
                    alall_expert = algates[:, 2] > 0
                    alpos_idx = (labels[alpos_expert] > 0).squeeze()
                    alneg_idx = (labels[alneg_expert] < 0).squeeze()
                    alnot0_idx = (labels[alall_expert] != 0).squeeze()
                    alis0_idx = (labels[alall_expert] == 0).squeeze()

                    avpos_expert = avgates[:, 0] > 0
                    avneg_expert = avgates[:, 1] > 0
                    avall_expert = avgates[:, 2] > 0
                    avpos_idx = (labels[avpos_expert] > 0).squeeze()
                    avneg_idx = (labels[avneg_expert] < 0).squeeze()
                    avnot0_idx = (labels[avall_expert] != 0).squeeze()
                    avis0_idx = (labels[avall_expert] == 0).squeeze()

                    vapos_expert = vagates[:, 0] > 0
                    vaneg_expert = vagates[:, 1] > 0
                    vaall_expert = vagates[:, 2] > 0
                    vapos_idx = (labels[vapos_expert] > 0).squeeze()
                    vaneg_idx = (labels[vaneg_expert] < 0).squeeze()
                    vanot0_idx = (labels[vaall_expert] != 0).squeeze()
                    vais0_idx = (labels[vaall_expert] == 0).squeeze()

                    vlpos_expert = vlgates[:, 0] > 0
                    vlneg_expert = vlgates[:, 1] > 0
                    vlall_expert = vlgates[:, 2] > 0
                    vlpos_idx = (labels[vlpos_expert] > 0).squeeze()
                    vlneg_idx = (labels[vlneg_expert] < 0).squeeze()
                    vlnot0_idx = (labels[vlall_expert] != 0).squeeze()
                    vlis0_idx = (labels[vlall_expert] == 0).squeeze()


                aldispatcher = SparseDispatcher(self.hp.expert_num, algates)
                ladispatcher = SparseDispatcher(self.hp.expert_num, lagates)
                avdispatcher = SparseDispatcher(self.hp.expert_num, avgates)
                lvdispatcher = SparseDispatcher(self.hp.expert_num, lvgates)
                vadispatcher = SparseDispatcher(self.hp.expert_num, vagates)
                vldispatcher = SparseDispatcher(self.hp.expert_num, vlgates)

                alexpert_inputs = aldispatcher.dispatch(al_c)
                # 所有专家非所有样本的输出
                alexpert_outputs = [self.con_pro[0][i](alexpert_inputs[i]) for i in
                                    range(self.hp.expert_num)]
                al_c = aldispatcher.combine(alexpert_outputs)

                avexpert_inputs = avdispatcher.dispatch(av_c)
                avexpert_outputs = [self.con_pro[1][i](avexpert_inputs[i]) for i in
                                    range(self.hp.expert_num)]

                av_c = aldispatcher.combine(avexpert_outputs)

                vaexpert_inputs = vadispatcher.dispatch(va_c)
                vaexpert_outputs = [self.con_pro[2][i](vaexpert_inputs[i]) for i in
                                    range(self.hp.expert_num)]
                va_c = vadispatcher.combine(vaexpert_outputs)

                vlexpert_inputs = vldispatcher.dispatch(vl_c)
                vlexpert_outputs = [self.con_pro[3][i](vlexpert_inputs[i]) for i in
                                    range(self.hp.expert_num)]
                vl_c = vldispatcher.combine(vlexpert_outputs)

                lvexpert_inputs = lvdispatcher.dispatch(lv_c)
                lvexpert_outputs = [self.con_pro[4][i](lvexpert_inputs[i]) for i in
                                    range(self.hp.expert_num)]
                lv_c = lvdispatcher.combine(lvexpert_outputs)

                laexpert_inputs = ladispatcher.dispatch(la_c)
                laexpert_outputs = [self.con_pro[4][i](laexpert_inputs[i]) for i in
                                    range(self.hp.expert_num)]
                la_c = ladispatcher.combine(laexpert_outputs)
                f_c = la_c + lv_c + al_c + av_c + vl_c + va_c
                keys = ["f","vl","lv","al","la","av","va",]

                for k in keys:
                    if  k == "f": #需要更新融合表征标签为统一标签，因为单模态标签更新需要
                        update_labels(self.hp,labels=labels.view(-1),mode="f")
                    if k != "f": # 若继承，则无论epoch为多少,stage=1 都需要且只更新单模态标签
                        if self.hp.nows > 1: # 继承与否则在epoch >2 的时候stage =0.5 or 1的时候更新
                            update_labels(self.hp, s=locals()[k + "_c"], f=f_c, mode=k)
                        elif self.hp.pretrain: #继承无论怎样stage=1则要更新标签
                            if self.hp.stage == 1:
                                update_labels(self.hp,s=locals()[k+"_c"],f=f_c,mode=k)

                    update_features(self.hp,locals()[k+"_c"],k) #求和正负特征留待下个epoch使用
                lv_polarity_labels = getattr(self.hp, "lv" + "_labels")
                la_polarity_labels = getattr(self.hp, "la" + "_labels")
                vl_polarity_labels = getattr(self.hp, "vl" + "_labels")
                va_polarity_labels = getattr(self.hp, "va" + "_labels")
                al_polarity_labels = getattr(self.hp, "al" + "_labels")
                av_polarity_labels = getattr(self.hp, "av" + "_labels")
                if (self.hp.polarity > 0 or self.hp.stage == 0.5) and self.training and self.hp.stage != 0:
                    criterion = self.weighted_loss
                    # 0和1的专家分别只预测积极和消极的样本
                    al_posexpert_y = self.classification[0](
                        alexpert_outputs[0][alpos_idx, ...] if len(alexpert_outputs[0][alpos_idx, ...].shape) != 3 else alexpert_outputs[0][alpos_idx, ...].squeeze(1))
                    al_negexpert_y = self.classification[0](
                        alexpert_outputs[1][alneg_idx, ...] if len(alexpert_outputs[1][alneg_idx, ...].shape) != 3 else alexpert_outputs[1][alneg_idx, ...].squeeze(1))
                    # 余下的1个专家预测所有选择样本
                    al_y = self.classification[0](alexpert_outputs[2])
                    al_loss = 0
                    if len(al_polarity_labels[alpos_expert][alpos_idx]) > 0 and self.equals(al_posexpert_y,al_polarity_labels[alpos_expert][alpos_idx]):
                        al_loss += criterion(al_posexpert_y,al_polarity_labels[alpos_expert][alpos_idx] if len(al_polarity_labels[alpos_expert][alpos_idx].shape) != 2 else al_polarity_labels[alpos_expert][alpos_idx].squeeze(1),"al",alpos_expert,alpos_idx)
                    if len(al_polarity_labels[alneg_expert][alneg_idx]) > 0 and self.equals(al_negexpert_y,al_polarity_labels[alneg_expert][alneg_idx]):
                        al_loss += criterion(al_negexpert_y,al_polarity_labels[alneg_expert][alneg_idx] if len(al_polarity_labels[alneg_expert][alneg_idx].shape) != 2 else al_polarity_labels[alneg_expert][alneg_idx].squeeze(1),"al",alneg_expert,alneg_idx)
                    if len(al_polarity_labels[alall_expert][alnot0_idx]) > 0 and self.equals(al_y[alnot0_idx], al_polarity_labels[
                        alall_expert][alnot0_idx]):
                        al_loss += criterion(al_y[alnot0_idx], al_polarity_labels[alall_expert][alnot0_idx] if len(
                            al_polarity_labels[alall_expert][alnot0_idx].shape) != 2 else
                        al_polarity_labels[alall_expert][alnot0_idx].squeeze(1),
                                             "al", alall_expert, alnot0_idx)
                    if len(al_polarity_labels[alall_expert][alis0_idx]) > 0 and self.equals(al_y[alis0_idx], al_polarity_labels[
                        alall_expert][alis0_idx]):
                        al_loss += self.hp.neu * criterion(al_y[alis0_idx], al_polarity_labels[alall_expert][alis0_idx] if len(
                            al_polarity_labels[alall_expert][alis0_idx].shape) != 2 else
                        al_polarity_labels[alall_expert][alis0_idx].squeeze(1),
                                             "al", alall_expert, alis0_idx)

                    av_posexpert_y = self.classification[1](
                        avexpert_outputs[0][avpos_idx, ...] if len(avexpert_outputs[0][avpos_idx, ...].shape) != 3 else avexpert_outputs[0][avpos_idx, ...].squeeze(1))
                    av_negexpert_y = self.classification[1](
                        avexpert_outputs[1][avneg_idx, ...] if len(avexpert_outputs[1][avneg_idx, ...].shape) != 3 else avexpert_outputs[1][avneg_idx, ...].squeeze(1))
                    av_y = self.classification[1](avexpert_outputs[2])
                    av_loss = 0
                    if len(av_polarity_labels[avpos_expert][avpos_idx]) > 0 and self.equals(av_posexpert_y, av_polarity_labels[avpos_expert][avpos_idx]):
                        av_loss += criterion(av_posexpert_y, av_polarity_labels[avpos_expert][avpos_idx] if len(av_polarity_labels[avpos_expert][avpos_idx].shape) != 2 else av_polarity_labels[avpos_expert][avpos_idx].squeeze(1),"av",avpos_expert,avpos_idx)
                    if len(av_polarity_labels[avneg_expert][avneg_idx]) > 0 and self.equals(av_negexpert_y, av_polarity_labels[avneg_expert][avneg_idx]):
                        av_loss = av_loss + self.hp.neg * criterion(av_negexpert_y,av_polarity_labels[avneg_expert][avneg_idx] if len(av_polarity_labels[avneg_expert][avneg_idx].shape) != 2 else av_polarity_labels[avneg_expert][avneg_idx].squeeze(1),"av",avneg_expert,avneg_idx)
                    if len(av_polarity_labels[avall_expert][avnot0_idx]) > 0 and self.equals(av_y[avnot0_idx], av_polarity_labels[
                        avall_expert][avnot0_idx]):
                        av_loss += criterion(av_y[avnot0_idx], av_polarity_labels[avall_expert][avnot0_idx] if len(
                            av_polarity_labels[avall_expert][avnot0_idx].shape) != 2 else
                        av_polarity_labels[avall_expert][avnot0_idx].squeeze(1),
                                             "av", avall_expert, avnot0_idx)
                    if len(av_polarity_labels[avall_expert][avis0_idx]) > 0 and self.equals(av_y[avis0_idx], av_polarity_labels[
                        avall_expert][avis0_idx]):
                        av_loss += self.hp.neu * criterion(av_y[avis0_idx], av_polarity_labels[avall_expert][avis0_idx] if len(
                            av_polarity_labels[avall_expert][avis0_idx].shape) != 2 else
                        av_polarity_labels[avall_expert][avis0_idx].squeeze(1),
                                                           "av", avall_expert, avis0_idx)


                    la_posexpert_y = self.classification[2](
                        laexpert_outputs[0][lapos_idx, ...] if len(laexpert_outputs[0][lapos_idx, ...].shape) != 3 else laexpert_outputs[0][lapos_idx, ...].squeeze(1))
                    la_negexpert_y = self.classification[2](
                        laexpert_outputs[1][laneg_idx, ...] if len(laexpert_outputs[1][laneg_idx, ...].shape) != 3 else laexpert_outputs[1][laneg_idx, ...].squeeze(1))
                    la_y = self.classification[2](laexpert_outputs[2])
                    la_loss = 0
                    if len(la_polarity_labels[lapos_expert][lapos_idx]) > 0 and self.equals(la_posexpert_y, la_polarity_labels[lapos_expert][lapos_idx]):
                        la_loss += criterion(la_posexpert_y, la_polarity_labels[lapos_expert][lapos_idx] if len(la_polarity_labels[lapos_expert][lapos_idx].shape) != 2 else la_polarity_labels[lapos_expert][lapos_idx].squeeze(1),"la",lapos_expert,lapos_idx)
                    if len( la_polarity_labels[laneg_expert][laneg_idx]) > 0 and self.equals(la_negexpert_y, la_polarity_labels[laneg_expert][laneg_idx]):
                        la_loss += self.hp.neg * criterion(la_negexpert_y,la_polarity_labels[laneg_expert][laneg_idx] if len(la_polarity_labels[laneg_expert][laneg_idx].shape) != 2 else la_polarity_labels[laneg_expert][laneg_idx].squeeze(1),"la",laneg_expert,laneg_idx)
                    if len(la_polarity_labels[laall_expert][lanot0_idx]) > 0 and self.equals(la_y[lanot0_idx],
                                                                                             la_polarity_labels[
                                                                                                 laall_expert][
                                                                                                 lanot0_idx]):
                        la_loss += criterion(la_y[lanot0_idx], la_polarity_labels[laall_expert][lanot0_idx] if len(
                            la_polarity_labels[laall_expert][lanot0_idx].shape) != 2 else
                        la_polarity_labels[laall_expert][lanot0_idx].squeeze(1),
                                             "la", laall_expert, lanot0_idx)
                    if len(la_polarity_labels[laall_expert][lais0_idx]) > 0 and self.equals(la_y[lais0_idx],
                                                                                            la_polarity_labels[
                                                                                                laall_expert][
                                                                                                lais0_idx]):
                        la_loss += self.hp.neu * criterion(la_y[lais0_idx],
                                                           la_polarity_labels[laall_expert][lais0_idx] if len(
                                                               la_polarity_labels[laall_expert][
                                                                   lais0_idx].shape) != 2 else
                                                           la_polarity_labels[laall_expert][lais0_idx].squeeze(1),
                                                           "la", laall_expert, lais0_idx)


                    lv_posexpert_y = self.classification[3](
                        lvexpert_outputs[0][lvpos_idx, ...] if len(lvexpert_outputs[0][lvpos_idx, ...].shape) != 3 else lvexpert_outputs[0][lvpos_idx, ...].squeeze(1))
                    lv_negexpert_y = self.classification[3](
                        lvexpert_outputs[1][lvneg_idx, ...] if len(lvexpert_outputs[1][lvneg_idx, ...].shape) != 3 else lvexpert_outputs[1][lvneg_idx, ...].squeeze(1))
                    lv_y = self.classification[3](lvexpert_outputs[2] )
                    lv_loss = 0
                    if len(lv_polarity_labels[lvpos_expert][lvpos_idx]) > 0 and self.equals(lv_posexpert_y, lv_polarity_labels[lvpos_expert][lvpos_idx]):
                        lv_loss += criterion(lv_posexpert_y, lv_polarity_labels[lvpos_expert][lvpos_idx] if len(lv_polarity_labels[lvpos_expert][lvpos_idx].shape) != 2 else lv_polarity_labels[lvpos_expert][lvpos_idx].squeeze(1),"lv",lvpos_expert,lvpos_idx)
                    if len(lv_polarity_labels[lvneg_expert][lvneg_idx]) > 0 and self.equals(lv_negexpert_y,lv_polarity_labels[lvneg_expert][lvneg_idx]):
                        lv_loss += self.hp.neg * criterion(lv_negexpert_y,lv_polarity_labels[lvneg_expert][lvneg_idx] if len(lv_polarity_labels[lvneg_expert][lvneg_idx].shape) != 2 else lv_polarity_labels[lvneg_expert][lvneg_idx].squeeze(1),"lv",lvneg_expert,lvneg_idx )
                    if len(lv_polarity_labels[lvall_expert][lvnot0_idx]) > 0 and self.equals(lv_y[lvnot0_idx],
                                                                                             lv_polarity_labels[
                                                                                                 lvall_expert][
                                                                                                 lvnot0_idx]):
                        lv_loss += criterion(lv_y[lvnot0_idx], lv_polarity_labels[lvall_expert][lvnot0_idx] if len(
                            lv_polarity_labels[lvall_expert][lvnot0_idx].shape) != 2 else
                        lv_polarity_labels[lvall_expert][lvnot0_idx].squeeze(1),
                                             "lv", lvall_expert, lvnot0_idx)
                    if len(lv_polarity_labels[lvall_expert][lvis0_idx]) > 0 and self.equals(lv_y[lvis0_idx],
                                                                                            lv_polarity_labels[
                                                                                                lvall_expert][
                                                                                                lvis0_idx]):
                        lv_loss += self.hp.neu * criterion(lv_y[lvis0_idx],
                                                           lv_polarity_labels[lvall_expert][lvis0_idx] if len(
                                                               lv_polarity_labels[lvall_expert][
                                                                   lvis0_idx].shape) != 2 else
                                                           lv_polarity_labels[lvall_expert][lvis0_idx].squeeze(1),
                                                           "lv", lvall_expert, lvis0_idx)


                    vl_posexpert_y = self.classification[4](
                        vlexpert_outputs[0][vlpos_idx, ...] if len(vlexpert_outputs[0][vlpos_idx, ...].shape) != 3 else vlexpert_outputs[0][vlpos_idx, ...].squeeze(1))
                    vl_negexpert_y = self.classification[4](
                        vlexpert_outputs[1][vlneg_idx, ...] if len(vlexpert_outputs[1][vlneg_idx, ...].shape) != 3 else vlexpert_outputs[1][vlneg_idx, ...].squeeze(1))
                    vl_y = self.classification[4](vlexpert_outputs[2])
                    vl_loss = 0
                    if len(vl_polarity_labels[vlpos_expert][vlpos_idx]) > 0 and self.equals(vl_posexpert_y, vl_polarity_labels[vlpos_expert][vlpos_idx]):
                        vl_loss += criterion(vl_posexpert_y, vl_polarity_labels[vlpos_expert][vlpos_idx] if len(vl_polarity_labels[vlpos_expert][vlpos_idx].shape) != 2 else vl_polarity_labels[vlpos_expert][vlpos_idx].squeeze(1),"vl",vlpos_expert,vlpos_idx)
                    if len(vl_polarity_labels[vlneg_expert][vlneg_idx]) > 0 and self.equals(vl_negexpert_y, vl_polarity_labels[vlneg_expert][vlneg_idx]):
                        vl_loss +=  self.hp.neg * criterion(vl_negexpert_y,vl_polarity_labels[vlneg_expert][vlneg_idx] if len(vl_polarity_labels[vlneg_expert][vlneg_idx].shape) != 2 else vl_polarity_labels[vlneg_expert][vlneg_idx].squeeze(1),"vl",vlneg_expert,vlneg_idx)
                    if len(vl_polarity_labels[vlall_expert][vlnot0_idx]) > 0 and self.equals(vl_y[vlnot0_idx],
                                                                                             vl_polarity_labels[
                                                                                                 vlall_expert][
                                                                                                 vlnot0_idx]):
                        vl_loss += criterion(vl_y[vlnot0_idx], vl_polarity_labels[vlall_expert][vlnot0_idx] if len(
                            vl_polarity_labels[vlall_expert][vlnot0_idx].shape) != 2 else
                        vl_polarity_labels[vlall_expert][vlnot0_idx].squeeze(1),
                                             "vl", vlall_expert, vlnot0_idx)
                    if len(vl_polarity_labels[vlall_expert][vlis0_idx]) > 0 and self.equals(vl_y[vlis0_idx],
                                                                                            vl_polarity_labels[
                                                                                                vlall_expert][
                                                                                                vlis0_idx]):
                        vl_loss += self.hp.neu * criterion(vl_y[vlis0_idx],
                                                           vl_polarity_labels[vlall_expert][vlis0_idx] if len(
                                                               vl_polarity_labels[vlall_expert][
                                                                   vlis0_idx].shape) != 2 else
                                                           vl_polarity_labels[vlall_expert][vlis0_idx].squeeze(1),
                                                           "vl", vlall_expert, vlis0_idx)

                    va_posexpert_y = self.classification[5](
                        vaexpert_outputs[0][vapos_idx, ...] if len(vaexpert_outputs[0][vapos_idx, ...].shape) != 3 else vaexpert_outputs[0][vapos_idx, ...].squeeze(1))
                    va_negexpert_y = self.classification[5](
                        vaexpert_outputs[1][vaneg_idx, ...] if len(vaexpert_outputs[1][vaneg_idx, ...].shape) != 3 else vaexpert_outputs[1][vaneg_idx, ...].squeeze(1))
                    va_y = self.classification[5](vaexpert_outputs[2])
                    va_loss = 0
                    if len(va_polarity_labels[vapos_expert][vapos_idx]) > 0:
                        va_loss += criterion(va_posexpert_y, va_polarity_labels[vapos_expert][vapos_idx] if len(va_polarity_labels[vapos_expert][vapos_idx].shape) != 2 else va_polarity_labels[vapos_expert][vapos_idx].squeeze(1),"va",vapos_expert,vapos_idx)
                    if len(va_polarity_labels[vaneg_expert][vaneg_idx]) > 0:
                        va_loss += self.hp.neg * criterion(va_negexpert_y,va_polarity_labels[vaneg_expert][vaneg_idx] if len(va_polarity_labels[vaneg_expert][vaneg_idx].shape) != 2 else va_polarity_labels[vaneg_expert][vaneg_idx].squeeze(1),"va",vaneg_expert,vaneg_idx)
                    if len(va_polarity_labels[vaall_expert][vanot0_idx]) > 0 and self.equals(va_y[vanot0_idx],
                                                                                             va_polarity_labels[
                                                                                                 vaall_expert][
                                                                                                 vanot0_idx]):
                        va_loss += criterion(va_y[vanot0_idx], va_polarity_labels[vaall_expert][vanot0_idx] if len(
                            va_polarity_labels[vaall_expert][vanot0_idx].shape) != 2 else
                        va_polarity_labels[vaall_expert][vanot0_idx].squeeze(1),
                                             "va", vaall_expert, vanot0_idx)
                    if len(va_polarity_labels[vaall_expert][vais0_idx]) > 0 and self.equals(va_y[vais0_idx],
                                                                                            va_polarity_labels[
                                                                                                vaall_expert][
                                                                                                vais0_idx]):
                        va_loss += self.hp.neu * criterion(va_y[vais0_idx],
                                                           va_polarity_labels[vaall_expert][vais0_idx] if len(
                                                               va_polarity_labels[vaall_expert][
                                                                   vais0_idx].shape) != 2 else
                                                           va_polarity_labels[vaall_expert][vais0_idx].squeeze(1),
                                                           "va", vaall_expert, vais0_idx)



                    if self.hp.stage == 0.5:
                        l_class = 0.5 * (va_loss + vl_loss) + 0.5 * (av_loss + al_loss) + 0.5*(la_loss + lv_loss)
                    else:
                        l_class = self.hp.polarity * (va_loss+vl_loss+av_loss+al_loss+la_loss+lv_loss)
                    # if l_class == 0 and self.hp.stage == 0.5:
                    #     l_class = torch.tensor(0)


                alaux = 0
                avaux = 0
                vlaux = 0
                vaaux = 0
                laaux = 0
                lvaux = 0
                aux_loss = 0
                if self.hp.aux > 0 and self.training and self.hp.stage == 1:
                    for i in range(self.hp.expert_num):
                        alaux = alaux + (len(alexpert_inputs[i]) / batch) * almE[i]
                        avaux = avaux + (len(avexpert_inputs[i]) / batch) * avmE[i]
                        vlaux = vlaux + (len(vlexpert_inputs[i]) / batch) * vlmE[i]
                        vaaux = vaaux + (len(vaexpert_inputs[i]) / batch) * vamE[i]
                        laaux = laaux + (len(laexpert_inputs[i]) / batch) * lamE[i]
                        lvaux = lvaux + (len(lvexpert_inputs[i]) / batch) * lvmE[i]
                        aux_loss = alaux * self.hp.aux + self.hp.aux * avaux + self.hp.aux * vlaux + self.hp.aux * vaaux + self.hp.aux * laaux + self.hp.aux * lvaux
            if self.training and self.hp.stage == 1 and max(self.hp.tv,self.hp.ta) > 0:
                l_con = self.joint_loss.compute(la_c,al_c,av_c,lv_c,vl_c,va_c,labels) + self.hp.va_in*self.ntxent_loss3.compute(va_c,labels) + self.hp.la_in*self.ntxent_loss2.compute(la_c,labels)

        # 中级融合，L,N,30
        # 3*30
        if self.hp.interaction == 'add':
            v = (vl_c + lv_c)
            l = (la_c + lv_c)
            a = (al_c + av_c)
        # 3,60
        elif self.hp.interaction == 'cat':
            v = torch.cat((vl_c,lv_c),dim = 1)
            l = torch.cat((la_c,lv_c),dim = 1)
            a = torch.cat((al_c,av_c),dim = 1)
        # 用于CPC的最终融合
        if self.hp.fin_interaction == 'add': # L,N,120
            info = self.pronabc(F.relu(self.prona1(a) + self.pronb1(v) + self.pronc1(l)) / 3)  # 有效信息,120
        elif self.hp.fin_interaction == 'cat': #L,N,120
            info = self.pronabc(torch.cat((v,a,l),dim= 1))

        # 对比学习部分,相同模态相同标签噪声的靠近,不同模态不同标签噪声远离
        # 只有info用于MMIM
        last_hs = self.proj3(info)  # torch.Size([32, 128])
        if self.hp.out == 'cat':  # 3*60 or 3*30 ,取决于interaction为cat or add -> cat后为180 or 90 ->
            output = self.out(torch.cat((v,a,l),dim=1))  # torch.Size([32, 1])
        else:   #  60 or 30 ->`
            output = self.out(v+a+l)
        return last_hs, output,l_con,sparse_aux_loss,aux_loss,l_class
    def equals(self,x,y):
        y = y if len(y.shape) != 2 else y.squeeze(1)
        if len(x) == y.shape[0]:
            return True
        else:
            return False
    def fix(self,x):
        for i in range(len(x)):
            xi = x[i]
            _,d = xi.sort()
            if d[2] == 2 or d[1] == 2:
                continue
            elif xi[d[2]] == 1:
                continue
            else:
                xi[2] = xi[d[1]] #把第二大的门控值赋给全能专家，保证他一直工作
                xi[d[1]] = 0
class CLUB(nn.Module):
    def __init__(self, hidden_size, activation='Tanh'):
        super(CLUB, self).__init__()
        try:
            self.activation = getattr(nn, activation)
        except:
            raise ValueError("Error: CLUB activation function not found in torch library")
        self.mlp_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.activation(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.activation(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, modal_a, modal_b, sample=False):
        mu, logvar = self.mlp_mu(modal_a), self.mlp_logvar(modal_a)  # (bs, hidden_size)
        batch_size = mu.size(0)
        pred = mu

        # pred b using a
        pred_tile = mu.unsqueeze(1).repeat(1, batch_size, 1)  # (bs, bs, emb_size)
        true_tile = pred.unsqueeze(0).repeat(batch_size, 1, 1)  # (bs, bs, emb_size)

        positive = - (mu - modal_b) ** 2 / 2. / torch.exp(logvar)
        negative = - torch.mean((true_tile - pred_tile) ** 2, dim=1) / 2. / torch.exp(logvar)

        lld = torch.mean(torch.sum(positive, -1))
        bound = torch.mean(torch.sum(positive, -1) - torch.sum(negative, -1))
        return lld, bound


class MMILB(nn.Module):  # 双模态表示 的 互信息下界
    """Compute the Modality Mutual Information Lower Bound (MMILB) given bimodal representations.
    Args:
        x_size (int): embedding size of input modality representation x
        y_size (int): embedding size of input modality representation y
        mid_activation(int): the activation function in the middle layer of MLP
        last_activation(int): the activation function in the last layer of MLP that outputs logvar
    """

    def __init__(self, x_size, y_size, mid_activation='ReLU', last_activation='Tanh'):
        super(MMILB, self).__init__()
        try:
            self.mid_activation = getattr(nn, mid_activation)
            self.last_activation = getattr(nn, last_activation)
        except:
            raise ValueError("Error: CLUB activation function not found in torch library")
        self.mlp_mu = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size),
        )
        self.entropy_prj = nn.Sequential(
            nn.Linear(y_size, y_size // 4),
            nn.Tanh()
        )

    def forward(self, x, y, labels=None, mem=None):
        """ Forward lld (gaussian prior) and entropy estimation, partially refers the implementation
        of https://github.com/Linear95/CLUB/blob/master/MI_DA/MNISTModel_DANN.py
            Args:
                x (Tensor): x in above equation, shape (bs, x_size)
                y (Tensor): y in above equation, shape (bs, y_size)
        """
        mu, logvar = self.mlp_mu(x), self.mlp_logvar(x)  # (bs, hidden_size)
        batch_size = mu.size(0)  # 32

        positive = -(mu - y) ** 2 / 2. / torch.exp(logvar)  # 负对数似然损失 # 32*16
        lld = torch.mean(torch.sum(positive, -1))  # tensor(-2.1866, grad_fn=<MeanBackward0>)

        # For Gaussian Distribution Estimation 高斯分布估计
        pos_y = neg_y = None
        H = 0.0
        sample_dict = {'pos': None, 'neg': None}

        if labels is not None:
            # store pos and neg samples
            y = self.entropy_prj(y)
            pos_y = y[labels.squeeze() > 0]  # 积极样本
            neg_y = y[labels.squeeze() < 0]  # 消极样本

            sample_dict['pos'] = pos_y
            sample_dict['neg'] = neg_y

            # estimate entropy
            if mem is not None and mem.get('pos', None) is not None:
                pos_history = mem['pos']
                neg_history = mem['neg']

                # Diagonal setting            
                # pos_all = torch.cat(pos_history + [pos_y], dim=0) # n_pos, emb
                # neg_all = torch.cat(neg_history + [neg_y], dim=0)
                # mu_pos = pos_all.mean(dim=0)
                # mu_neg = neg_all.mean(dim=0)

                # sigma_pos = torch.mean(pos_all ** 2, dim = 0) - mu_pos ** 2 # (embed)
                # sigma_neg = torch.mean(neg_all ** 2, dim = 0) - mu_neg ** 2 # (embed)
                # H = 0.25 * (torch.sum(torch.log(sigma_pos)) + torch.sum(torch.log(sigma_neg)))

                # compute the entire co-variance matrix,结合历史的积极和消极样本，若历史为空则H=0（要在足够大的采样批量上进行）
                pos_all = torch.cat(pos_history + [pos_y], dim=0)  # n_pos, emb
                neg_all = torch.cat(neg_history + [neg_y], dim=0)
                mu_pos = pos_all.mean(dim=0)
                mu_neg = neg_all.mean(dim=0)
                sigma_pos = torch.mean(torch.bmm((pos_all - mu_pos).unsqueeze(-1), (pos_all - mu_pos).unsqueeze(1)),
                                       dim=0)  # mycal: t1 = pos_all * pos_all;t2 = t1.mean(dim=0),t3 = (t2 - mu_pos*mu_pos)
                sigma_neg = torch.mean(torch.bmm((neg_all - mu_neg).unsqueeze(-1), (neg_all - mu_neg).unsqueeze(1)),
                                       dim=0)

                a = 17.0795
                if min(torch.det(sigma_pos), torch.det(sigma_neg)) > 0:
                    H = 0.25 * (torch.logdet(sigma_pos) + torch.logdet(sigma_neg))  # 公式(8)

        return lld, sample_dict, H


class CPC(nn.Module):  # 对比预测编码（可以更换！！！）
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.

        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """

    # 768,128
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        # x是：t a v     y是融合后的
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)  # 激活层的激活函数：tanh
        if n_layers == 1:  # 进
            self.net = nn.Linear(
                in_features=y_size,
                out_features=x_size
            )
        else:
            net = []
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Linear(self.y_size, self.x_size))
                    net.append(self.activation())
                else:
                    net.append(nn.Linear(self.x_size, self.x_size))
            self.net = nn.Sequential(*net)

    def forward(self, x, y):
        """Calulate the score
            公式11
            eg： nce_t = self.cpc_zt(text, fusion)  # 3.4660
            x: torch.Size([32, 768]) single
            y: torch.Size([32, 128]) fusion    torch.Size([32, 180])(transformer)
        """
        # import ipdb;ipdb.set_trace()
        x_pred = self.net(y)  # bs, emb_size torch.Size([32, 768])
        # 从融合结果y生成的G(Z) 实际是hm的反向预测值  这个net是G（反向传播网络）

        # normalize to unit sphere  归一化
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)  # G 归一化  公式10 torch.Size([32, 768])
        x = x / x.norm(dim=1, keepdim=True)  # hm 归一化  公式10  torch.Size([32, 768])

        pos = torch.sum(x * x_pred, dim=-1)  # 没有exp是因为被log抵消了，正样本相似度，即当前样本的相似度 bs shape:32   公式10: 得到 s(hm,Z)
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)  # bs
        nce = -(pos - neg).mean()
        return nce


class RNNEncoder(nn.Module):  # 视频和音频的特征提取   也可以换！！！
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.bidirectional = bidirectional
        #
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                           batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear((2 if bidirectional else 1) * hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        eg: self.visual_enc(visual, v_len) # torch.Size([134, 32, 5]) ,tensor:32
        '''
        lengths = lengths.to(torch.int64)  # tensor:32
        bs = x.size(0)  # bs：batch_size  134   将x的第一个size赋值给bs

        packed_sequence = pack_padded_sequence(x, lengths, enforce_sorted=False)
        # 将序列送给 RNN 进行处理之前，需要采用 pack_padded_sequence 进行压缩，压缩掉无效的填充值
        _, final_states = self.rnn(packed_sequence)

        if self.bidirectional:  # 是否使用双向RNN
            h = self.dropout(torch.cat((final_states[0][0], final_states[0][1]), dim=-1))
        else:  # 这里
            h = self.dropout(final_states[0].squeeze())  # torch.Size([32, 8])
        y_1 = self.linear_1(h)
        return y_1  # 32*16
