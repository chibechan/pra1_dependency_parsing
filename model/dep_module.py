import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DepModule(nn.Module):
    def __init__(self, hidden_size, dep_dim):
        super(DepModule, self).__init__()
        self.dep_dim = dep_dim
        self.U_h = nn.Linear(hidden_size, dep_dim, bias=False)
        self.W_h = nn.Linear(hidden_size, dep_dim, bias=False)
        self.v_h = nn.Linear(dep_dim, 1, bias=False)
        self.u_h = nn.Linear(dep_dim, 37, bias=False)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        self.tanh = nn.Tanh()

    def forward(self, pooled_output, s_mask, dep=None, mapping_matrix=None,
                label=None, dep_list=None, train=True, ud=True, alpha=1):
        '''
        input:
            pooled_output: BERTの出力
            s_mask: Dependencyに関係ないところをmaskする隣接行列
            dep: Dependencyを隣接行列形式で表したもの, predictionの時はNone
            mapping_matrix: 形態素を基本句or文節に変換するためのmatrix
            label: それぞれの係り受けのラベル, predictionのときはNone
                   (ただし、labelのpredictionも行うときはTrue)
            dep_list: Dependencyをリスト形式で表したもの
            　　　　　　predictionのときはNone
            train: trainかpredictionか
            ud: universal dependencyのとき（つまりlabelあり)のときはtrue
        output:
            loss: depndencyのheadを予測する問題とlabelを予測する問題のlossを足し合わせた
            dependnecy_prediction: headのprediction。行列形式で答えを返す
            label_prediction: labelのprediction。list形式で答えを解す。
        '''
        target_vectors = torch.matmul(mapping_matrix, pooled_output)  # batch * Seq_len * hidden

        # Labelなしのparsingを行う
        batch_size = target_vectors.shape[0]
        seq_len = target_vectors.shape[1]
        t_i = self.W_h(target_vectors).view(
            batch_size, seq_len, 1, self.dep_dim)  # batch * seq_len * 1 * dep_dim
        t_j = self.U_h(target_vectors).view(
            batch_size, 1, seq_len, self.dep_dim)  # bath * 1 * seq_len * dep_dim
        s_ij = self.v_h(self.tanh(t_j + t_i))  # batch * Seq_len * seq_len * 1
        s_ij = s_ij.view(-1, s_ij.shape[1], s_ij.shape[1])  # batch * seq_len * seq_len
        s_ij += s_mask  # sを関係する箇所以外-1.0e-9でうめる
        s_ij = self.logsoftmax(s_ij)

        # trainならlossの計算
        if (train):
            loss = -1*torch.sum(s_ij * dep)/torch.sum(dep)  # cross entropyの計算
        else:
            loss = 0
        # labelなしのdependencyのprediction
        prediction = F.one_hot(torch.argmax(s_ij, dim=2),
                               num_classes=s_ij.shape[1])

        # labelなしのときはここで終わり
        if ud is False:
            return loss, prediction, None

        # labelありかつtrain
        elif train is True:  # labelあり+訓練
            index_1 = []
            for i in range(dep_list.shape[0]):
                index_1 += [i]*dep_list.shape[-1]
            index_1 = torch.LongTensor(index_1)
            index_2 = torch.LongTensor(
                list(range(dep_list.shape[-1]))*dep_list.shape[0])
            index_3 = dep_list.view(-1)
            l_i = (t_i+t_j)[(index_1, index_2, index_3)].reshape(
                dep_list.shape[0], dep_list.shape[-1], self.dep_dim)  # batch*Seq_len*dep_dim
            l_i = self.logsoftmax(self.u_h(self.tanh(l_i)))

            # labelのone hot encoding
            one_hot = torch.nn.functional.one_hot(label, num_classes=38)  # batch*Seq_len*38
            one_hot = np.delete(np.array(one_hot.cpu()), 37, -1)  # batch*Seq_len*37
            one_hot_tensor = torch.LongTensor(one_hot).to(DEVICE)

            loss += -alpha*torch.sum(l_i * one_hot_tensor) \
                                 / torch.sum(one_hot_tensor)  # cross entropyの計算
            label_prediction = torch.argmax(l_i, dim=2)

            return loss, prediction, label_prediction

        # labelありかつprediction
        elif train is False:  # labelあり+訓練
            # 推論時はpredict depを使う
            pdep_list = torch.argmax(prediction, dim=2)
            # labelに関係あるt_i+t_jだけ取り出す
            index_1 = []
            for i in range(pdep_list.shape[0]):
                index_1 += [i]*pdep_list.shape[-1]
            index_1 = torch.LongTensor(index_1)
            index_2 = torch.LongTensor(
                list(range(pdep_list.shape[-1]))*pdep_list.shape[0])
            index_3 = pdep_list.view(-1)
            l_i = (t_i+t_j)[(index_1, index_2, index_3)].reshape(
                pdep_list.shape[0], pdep_list.shape[-1], self.dep_dim)  # batch*Seq_len*dep_dim
            l_i = self.logsoftmax(self.u_h(self.tanh(l_i)))

            label_prediction = torch.argmax(l_i, dim=2)

            return loss, prediction, label_prediction
