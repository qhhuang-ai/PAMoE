import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet, FusionTrans
from torch.distributions.normal import Normal
from transformers import BertModel, BertConfig
from utils.selfMM import update_features,update_centers,update_labels

class PAMoE(nn.Module):
    def __init__(self, hp):

        # Base Encoders
        super().__init__()

        self.add_va = hp.add_va
        # 768
        hp.d_tout = hp.d_tin
        self.hp = hp
        self.text_enc = LanguageEmbeddingLayer(hp)  # BERT Encoder
        # 路由权重定义
        if self.hp.expert == 'RNN':
            self.v_router = nn.Linear(hp.vf * hp.vl,hp.expert_num,bias=False)
            self.a_router = nn.Linear(hp.af * hp.al, hp.expert_num,bias=False)
        if self.hp.noise_aux > 0:
            self.vnoise = nn.Linear(hp.vf * hp.vl,hp.expert_num,bias=False)
            self.anoise = nn.Linear(hp.af * hp.al, hp.expert_num,bias=False)
            self.softplus = nn.Softplus()
            self.softmax = nn.Softmax(1)
            self.register_buffer("mean", torch.tensor([0.0]))
            self.register_buffer("std", torch.tensor([1.0]))
        if self.hp.expert == 'RNN':
            self.visual_enc = nn.ModuleList([
                RNNEncoder(  # 视频特征提取
                    in_size=hp.d_vin,
                    hidden_size=hp.d_vh,
                    out_size=hp.d_vout,
                    num_layers=hp.n_layer,
                    dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
                    bidirectional=hp.bidirectional
                )
                for i in range(self.hp.expert_num)])
            self.acoustic_enc = nn.ModuleList([
                RNNEncoder(  # 音频特征提取
                    in_size=hp.d_ain,
                    hidden_size=hp.d_ah,
                    out_size=hp.d_aout,
                    num_layers=hp.n_layer,
                    dropout=hp.dropout_a if hp.n_layer or hp.a_in > 16 else 0.0,
                    bidirectional=hp.bidirectional
                )
                for i in range(self.hp.expert_num)])
        else:
            self.visual_enc = RNNEncoder(  # 视频特征提取
                in_size=hp.d_vin,
                hidden_size=hp.d_vh,
                out_size=hp.d_vout,
                num_layers=hp.n_layer,
                dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
                bidirectional=hp.bidirectional
            )
            self.acoustic_enc = RNNEncoder(  # 音频特征提取
                in_size=hp.d_ain,
                hidden_size=hp.d_ah,
                out_size=hp.d_aout,
                num_layers=hp.n_layer,
                dropout=hp.dropout_a if hp.n_layer or hp.a_in > 16 else 0.0,
                bidirectional=hp.bidirectional
            )

        self.mi_tv = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_vout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )

        self.mi_ta = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_aout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )

        if hp.add_va:
            self.mi_va = MMILB(
                x_size=hp.d_vout,
                y_size=hp.d_aout,
                mid_activation=hp.mmilb_mid_activation,
                last_activation=hp.mmilb_last_activation
            )

        dim_sum = hp.d_aout + hp.d_vout + hp.d_tout  # 计算所有模态输出后的维度和 用于后期融合操作

        # CPC MI bound
        self.cpc_zt = CPC(
            x_size=hp.d_tout,  # to be predicted  各个模态特征提取后得到的维度
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_zv = CPC(
            x_size=hp.d_vout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_za = CPC(
            x_size=hp.d_aout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )

        # Trimodal Settings   三模态融合
        self.fusion_prj = SubNet(
            in_size=dim_sum,  # 三个单模态输出维度和
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,  # 最终分类类别
            dropout=hp.dropout_prj
        )
        # 用MULT融合
        self.fusion_trans = FusionTrans(
            hp,
            n_class=hp.n_class,  # 最终分类类别
        )
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
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
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2,mode='v'):
        # 均值or方差的计算
        route = getattr(self,mode+'_router')
        if self.hp.noise_aux > 0:
            noise = route = getattr(self,mode+'noise')
        clean_logits = route(x)
        # 一个batch中每个专家出现概率和
        mE = clean_logits.mean(dim=0)
        if self.noise_aux > 0 and train:
            raw_noise_stddev = noise(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.hp.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        # 把非tok的置为0
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noise_aux > 0 and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load,mE
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
    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, ys, y=None,
                mem=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        sentences: torch.Size([0, 32])
        a: torch.Size([134, 32, 5])
        v: torch.Size([161, 32, 20])
        For Bert input, the length of text is "seq_len + 2"
        """
        # 1. 三个模态 分别 进行特征提取
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type,
                                 bert_sent_mask)  # 32*50*768 (batch_size, seq_len, emb_size)
        batch,_,_ = enc_word.shape
        text_trans = enc_word.transpose(0, 1)  # torch.Size([50, 32, 768]) (seq_len, batch_size,emb_size)
        vision_trans = visual
        audio_trans = acoustic

        text = enc_word[:, 0, :]  # 取首个时刻，在互信息和CPC中使用 32*768 (batch_size, emb_size)
        sparse_aux_loss = 0
        aux_loss = 0
        if self.hp.expert == 'RNN':
            agates, aload,amE = self.noisy_top_k_gating(acoustic.permute(1,0,2).reshape(batch,-1), self.training,mode='a')
            aimportance = agates.sum(0)

            vgates, vload,vmE = self.noisy_top_k_gating(visual.permute(1,0,2).reshape(batch, -1), self.training, mode='v')
            vimportance = vgates.sum(0)
            if self.hp.noise_aux > 0 and self.hp.stage == 1 and self.training:
                aauxloss = self.cv_squared(aimportance) + self.cv_squared(aload)
                vauxloss = self.cv_squared(vimportance) + self.cv_squared(vload)
                sparse_aux_loss = self.hp.noise_aux * aauxloss + self.hp.noise_aux * vauxloss
            adispatcher = SparseDispatcher(self.num_experts, agates)
            vdispatcher = SparseDispatcher(self.num_experts, vgates)

            aexpert_inputs = adispatcher.dispatch(acoustic.view(batch,self.hp.al,-1))
            aaux = 0
            vaux = 0
            aux_loss = 0
            aexpert_outputs = [self.acoustic_enc[i](aexpert_inputs[i].view(self.hp.al,batch,-1)) for i in range(self.hp.expert_num)]
            acoustic = adispatcher.combine(aexpert_outputs)

            vexpert_inputs = vdispatcher.dispatch(visual.view(batch,self.hp.vl,-1))
            vexpert_outputs = [self.visual_enc[i](vexpert_inputs[i].view(self.hp.vl, batch, -1)) for i in
                               range(self.hp.expert_num)]
            visual = vdispatcher.combine(vexpert_outputs)
            if self.hp.aux > 0 and self.hp.stage == 1 and self.training:
                for i in range(self.hp.expert_num):
                    aaux = aaux +  (len(aexpert_inputs[i]) / batch) * amE[i]
                    vaux = vaux +  (len(vexpert_inputs[i]) / batch) * vmE[i]
                    aux_loss = aaux * self.hp.aux + self.hp.aux * vaux
        else:
            acoustic = self.acoustic_enc(acoustic, a_len)  # 同样只有最后一个时间步的 32*16
            visual = self.visual_enc(visual, v_len)  # 32*16 同样只有最后一个时间步的 32*16

        if y is not None:
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual, labels=y, mem=mem['tv'])
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic, labels=y, mem=mem['ta'])
            # for ablation use
            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic, labels=y, mem=mem['va'])
        else:  # 默认进这
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual)  # mi_tv 模态互信息,lld为负对数似然
            # lld_tv:-2.1866  tv_pn:{'pos': None, 'neg': None}  H_tv:0.0
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic)
            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic)

        # Linear proj and pred
        # text:32*769   acoustic,visual:32*16   ->  cat后：[32, 801]
        # fusion, preds = self.fusion_prj(torch.cat([text, acoustic, visual], dim=1))
        # 32*128  32*1

        fusion, preds, l_con,lsp,laux,l_class = self.fusion_trans(text_trans, audio_trans, vision_trans, ys)
        # torch.Size([32, 180]) torch.Size([32, 1])
        if self.hp.expert != 'RNN':
            aux_loss = laux
            sparse_aux_loss = lsp
        nce_t = self.cpc_zt(text, fusion)  # 3.4660
        nce_v = self.cpc_zv(visual, fusion)  # 3.4625
        nce_a = self.cpc_za(acoustic, fusion)  # 3.4933

        nce = nce_t + nce_v + nce_a  # 10.4218  CPC loss

        pn_dic = {'tv': tv_pn, 'ta': ta_pn, 'va': va_pn if self.add_va else None}
        # {'tv': {'pos': None, 'neg': None}, 'ta': {'pos': None, 'neg': None}, 'va': None}
        lld = lld_tv + lld_ta + (lld_va if self.add_va else 0.0)  # -5.8927
        H = H_tv + H_ta + (H_va if self.add_va else 0.0)

        return lld, nce, preds, pn_dic, H, aux_loss, l_con, sparse_aux_loss,l_class

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

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
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

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
