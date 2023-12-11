import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import pandas as pd
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from utils.eval_metrics import *
from utils.tools import *
from model import PAMoE
from myutil.Plot import plot
from utils.selfMM import update_features,update_centers,update_labels
# torch.autograd.set_detect_anomaly(True)
torch.autograd.set_detect_anomaly(False)
class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None,
                 pretrained_emb=None):
        self.now_time = time.strftime("_%Y%m%d_%H%M", time.localtime())
        gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        gpu_ids = [int(i) for i in gpus]
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp' + hyp_params.no)
        memory_gpu = [int(x.split()[2]) for x in open('tmp' + hyp_params.no, 'r').readlines()]
        os.system("rm tmp" + hyp_params.no)
        memory = [memory_gpu[i] for i in gpu_ids]
        max_memory = max(memory)
        while True:
            if max_memory > 21000:
                break
            else:
                time.sleep(5)
                os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
                memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
                os.system("rm tmp"+ hyp_params.no)
                memory = [memory_gpu[i] for i in gpu_ids]
                max_memory = max(memory)
        for idx, i in enumerate(gpu_ids):
            if memory_gpu[i] == max_memory:
                self.gpu_id = idx
                break
        torch.cuda.set_device(self.gpu_id)
        setattr(hyp_params, "gpu_id", self.gpu_id)
        self.hp = hp = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.seed = hyp_params.seed
        self.is_train = is_train
        self.model = model
        self.name = hyp_params.dataset + self.now_time
        # Training hyperarams
        self.alpha = hp.alpha
        self.beta = hp.beta
        self.y_list = {'train_mae': [], 'valid_mae': [], 'test_mae': []}
        self.labels = [['train_mae', 'valid_mae', 'test_mae'], ['train_loss']]
        self.y_list1 = {'train_loss': []}
        self.img = './log/' + self.now_time + ""
        self.update_batch = hp.update_batch

        # initialize the model
        if model is None:
            self.model = model = PAMoE(hp)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            model = model.cuda()
        else:
            self.device = torch.device("cpu")

        # criterion
        if self.hp.dataset == "ur_funny":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        else:  # mosi and mosei are regression datasets
            self.criterion = criterion = nn.L1Loss(reduction="mean")

        # optimizer
        self.optimizer = {}

        if self.is_train:
            mmilb_param = []
            main_param = []
            bert_param = []

            for name, p in model.named_parameters():
                # print(name)
                if p.requires_grad:
                    if 'bert' in name:
                        bert_param.append(p)
                    elif 'mi' in name:
                        mmilb_param.append(p)
                    else:
                        main_param.append(p)

                for p in (mmilb_param + main_param):
                    if p.dim() > 1:  # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                        nn.init.xavier_normal_(p)

        self.optimizer_mmilb = getattr(torch.optim, self.hp.optim)(
            mmilb_param, lr=self.hp.lr_mmilb, weight_decay=hp.weight_decay_club)

        optimizer_main_group = [
            {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert},
            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
        ]

        self.optimizer_main = getattr(torch.optim, self.hp.optim)(
            optimizer_main_group
        )

        self.scheduler_mmilb = ReduceLROnPlateau(self.optimizer_mmilb, mode='min', patience=hp.when, factor=0.5,
                                                 verbose=True)
        self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, mode='min', patience=hp.when, factor=0.5,
                                                verbose=True)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        optimizer_mmilb = self.optimizer_mmilb
        optimizer_main = self.optimizer_main

        scheduler_mmilb = self.scheduler_mmilb
        scheduler_main = self.scheduler_main

        # criterion for downstream task
        criterion = self.criterion

        # entropy estimate interval
        # mem_size = 256
        # 用于建立混合高斯模型所选的历史样本批次容量
        mem_size = self.hp.mem

        def train(model, optimizer, criterion, stage=1):
            setattr(self.hp,'stage',stage)
            epoch_loss = 0

            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0
            nce_loss = 0.0
            ba_loss = 0.0
            start_time = time.time()
            l1_loss = 0.0
            left_batch = self.update_batch #1
            # cpc中的正负对索引
            mem_pos_tv = []
            mem_neg_tv = []
            mem_pos_ta = []
            mem_neg_ta = []
            if self.hp.add_va:
                mem_pos_va = []
                mem_neg_va = []

            mae_loss = 0
            all_loss = 0
            # y_loss = 0
            for i_batch, batch_data in enumerate(tqdm(self.train_loader)):
                text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data
                # (0,32) (176,32,20)  tensor:32 (147,32,5) tensor:32 32 32 (32,50) (32,50) 32 32

                # for mosei we only use 50% dataset in stage 1
                if self.hp.dataset == "mosei":
                    if stage == 0 and i_batch / len(self.train_loader) >= 0.5:
                        break
                    elif stage == 0 and i_batch / len(self.train_loader) >= 0.5:
                        break
                model.zero_grad()

                text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                    text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), \
                    bert_sent_type.cuda(), bert_sent_mask.cuda()
                if self.hp.dataset == "ur_funny":
                    y = y.squeeze()

                batch_size = y.size(0)
                ys = y.clone().detach().cpu()
                if stage == 0:  # Neg-lld  默认stage=1
                    y = None
                    mem = None
                elif stage == 0.5: # for polarity
                    y = None
                    mem = None
                elif stage == 1 and i_batch >= mem_size:  # TASK+BA+CPC  memory
                    mem = {'tv': {'pos': mem_pos_tv, 'neg': mem_neg_tv},
                           'ta': {'pos': mem_pos_ta, 'neg': mem_neg_ta},
                           'va': {'pos': mem_pos_va, 'neg': mem_neg_va} if self.hp.add_va else None}
                else:
                    mem = {'tv': None, 'ta': None, 'va': None}
                # 训练模型，得到（极大似然loss，平均绝对误差loss，预测值，，）熵H:首个batch为0.0
                lld, nce, preds, pn_dic, H,aux_loss, l_con, sparse_aux_loss,l_class = model(text, visual, audio, vlens, alens,
                                                   bert_sent, bert_sent_type, bert_sent_mask, ys,y, mem)

                if stage == 1:  # TASK+BA+CPC
                    y_loss = criterion(preds, y)  # mosei/mosi:L1Loss
                    l1_loss += y_loss * y.shape[0]
                    # update memory
                    if len(mem_pos_tv) < mem_size:
                        mem_pos_tv.append(pn_dic['tv']['pos'].detach())
                        mem_neg_tv.append(pn_dic['tv']['neg'].detach())
                        mem_pos_ta.append(pn_dic['ta']['pos'].detach())
                        mem_neg_ta.append(pn_dic['ta']['neg'].detach())
                        if self.hp.add_va:
                            mem_pos_va.append(pn_dic['va']['pos'].detach())
                            mem_neg_va.append(pn_dic['va']['neg'].detach())
                    else:  # memory is full! replace the oldest with the newest data
                        oldest = i_batch % mem_size
                        mem_pos_tv[oldest] = pn_dic['tv']['pos'].detach()
                        mem_neg_tv[oldest] = pn_dic['tv']['neg'].detach()
                        mem_pos_ta[oldest] = pn_dic['ta']['pos'].detach()
                        mem_neg_ta[oldest] = pn_dic['ta']['neg'].detach()

                        if self.hp.add_va:
                            mem_pos_va[oldest] = pn_dic['va']['pos'].detach()
                            mem_neg_va[oldest] = pn_dic['va']['neg'].detach()

                    if self.hp.contrast:
                        loss = y_loss + self.alpha * nce - self.beta * lld  + aux_loss + l_con + sparse_aux_loss + l_class   # 公式 14

                    else:
                        loss = y_loss
                    if i_batch > mem_size:
                        loss -= self.beta * H

                    loss.backward()
                elif stage == 0.5:
                    loss = l_class
                    loss.backward()
                elif stage == 0:
                    # maximize likelihood equals minimize neg-likelihood
                    loss = -lld
                    loss.backward()
                else:
                    raise ValueError('stage index can either be 0 or 1 or 0.5')

                left_batch -= 1
                if left_batch == 0:
                    left_batch = self.update_batch
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip) # 将所有参数的梯度缩放到指定的范围内
                    optimizer.step()
                if type(loss) == int:
                    loss = torch.tensor(loss)
                all_loss = all_loss + loss.item() * batch_size

                proc_loss = proc_loss + loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size
                nce_loss += nce.item() * batch_size  # CPC loss
                ba_loss += (-H - lld) * batch_size  # BA loss

                if i_batch % self.hp.log_interval == 0 and i_batch > 0:
                    avg_loss = proc_loss / proc_size
                    elapsed_time = time.time() - start_time
                    avg_nce = nce_loss / proc_size  #
                    avg_ba = ba_loss / proc_size
                    print(
                        'Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss ({}) {:5.4f} | NCE {:.3f} | BA {:.4f}'.
                            format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval,
                                   'TASK+BA+CPC' if stage == 1 else 'Neg-lld',
                                   avg_loss, avg_nce, avg_ba))
                    proc_loss, proc_size = 0, 0
                    nce_loss = 0.0
                    ba_loss = 0.0
                    start_time = time.time()
            if stage == 1:
                print(
                    f'Epoch {epoch} |  MAE Loss ({(l1_loss / self.hp.n_train)})  | ALL Loss {all_loss / self.hp.n_train}')
                self.y_list['train_mae'].append((l1_loss  / self.hp.n_train).item())
                self.y_list1['train_loss'].append(epoch_loss / self.hp.n_train)
            return epoch_loss / self.hp.n_train

        def evaluate(model, criterion, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0 # 后面变成L1loss了
            total_l1_loss = 0.0

            results = []
            truths = []

            # 验证模型
            with torch.no_grad():
                for batch in loader:
                    text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch


                    text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
                    lengths = lengths.cuda()
                    bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()
                    if self.hp.dataset == 'iemocap':
                        y = y.long()

                    if self.hp.dataset == 'ur_funny':
                        y = y.squeeze()

                    batch_size = lengths.size(0)  # bert_sent in size (bs, seq_len, emb_size)
                    ys = y.clone().detach().cpu()
                    # we don't need lld and bound anymore
                    _, _, preds, _, _,aux_loss, l_con, sparse_aux_loss,l_class = model(text, vision, audio, vlens, alens, bert_sent, bert_sent_type,
                                              bert_sent_mask,ys)

                    if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti'] and test:
                        criterion = nn.L1Loss()  # 任务误差，MAE

                    total_loss += criterion(preds, y).item() * batch_size

                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(preds)
                    truths.append(y)

            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)
            if test:
                self.y_list['test_mae'].append(avg_loss.item() if type(avg_loss) != float else avg_loss)
            else:
                self.y_list['valid_mae'].append(avg_loss.item()  if type(avg_loss) != float else avg_loss)
            results = torch.cat(results)  # torch.Size([229, 1])
            truths = torch.cat(truths)  # torch.Size([229, 1])

            #test
            # print(truths)
            # McNemar_test = {
            #     'results': results.cpu().squeeze(1),
            #     'truths': truths.cpu().squeeze(1)
            # }
            # data_frame = pd.DataFrame(data=McNemar_test)
            # data_frame.to_csv('TeFNA_mosi_MTest.csv')


            return avg_loss, results, truths
        mymae = []
        ress = []
        best_res = []
        best_valid = 1e8
        best_mae = 1e8
        patience = self.hp.patience

        for epoch in range(1, self.hp.num_epochs + 1):
            setattr(self.hp,"nows",epoch)
            start = time.time()

            self.epoch = epoch
            # 每个epoch中，先只最大化负对数似然及相关参数，在最小化所有损失
            # maximize likelihood
            if self.hp.contrast:
                train_loss = train(model, optimizer_mmilb, criterion, 0)
            train_loss = train(model, optimizer_mmilb, criterion, 0.5)
            # minimize all losses left
            if self.hp.pretrain:
                update_centers(self.hp)
            train_loss = train(model, optimizer_main, criterion, 1)
            # L1loss
            update_centers(self.hp)
            val_loss, _, _ = evaluate(model, criterion, test=False)
            test_loss, results, truths = evaluate(model, criterion, test=True)
            ress.append(eval_mosei_senti(results, truths, True))
            mymae.append(test_loss)
            end = time.time()
            duration = end - start
            scheduler_main.step(val_loss)  # Decay learning rate by validation loss

            # validation F1
            # print("-" * 50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration,
                                                                                                   val_loss, test_loss))
            # print("-" * 50)
            # val最小且test最小
            if val_loss < best_valid:
                # update best validation
                patience = self.hp.patience
                best_valid = val_loss
                # for ur_funny we don't care about
                if self.hp.dataset == "ur_funny":
                    eval_humor(results, truths, True)
                elif test_loss < best_mae:
                    best_epoch = epoch
                    best_mae = test_loss
                    # 验证模型
                    if self.hp.dataset in ["mosei_senti", "mosei",'mosi']:
                        res = eval_mosei_senti(results, truths, True)
                    elif self.hp.dataset == 'mosi':
                        res = eval_mosi(results, truths, True)
                    elif self.hp.dataset == 'iemocap':
                        eval_iemocap(results, truths)

                    # print(res['to_exl'])

                    best_results = results
                    best_truths = truths
                    now_time = time.strftime("_%Y%m%d_%H%M", time.localtime())
                    mae = str(best_mae).split('.')[0] + '.' + str(best_mae).split('.')[1][:5]

            else:
                patience -= 1
                if patience == 0:
                    break

        print(f'Best epoch: {best_epoch}')
        # plot('mae loss', self.img + 'mae.png', [i for i in range(1, self.epoch + 1)], self.y_list, 'epoch', 'mae')
        # plot('train loss', self.img + 'train.png', [i for i in range(1, self.epoch + 1)], self.y_list1, 'epoch',
        #      'train_loss')
        best_res = eval_mosei_senti(best_results, best_truths, True, self.seed, self.name)
        return best_mae,mymae,ress,best_res
        sys.stdout.flush()
