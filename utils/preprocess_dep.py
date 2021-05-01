import pandas as pd
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
from utils.mapping import make_untokenize_mapping_lists
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INF = 1.0e9


class DepDataset(Dataset):
    '''
    input:
        df: dataframe
        vocab_path: NICT BERTのvocabへのpath
        max_len: BERTの最大入力長 UDでは320
        BEP: BPEありかなしか
    getitemで返す要素:
        inputs_lst: 形態素分割された文章をvocabにより数値に変換したリスト
        masks: ↑のlistにおいて、要素が入っている位置を1、要素がない場所を0としたmask
        segment: 今回は[SEP]で切り分けないので全部0でOK
        paths: この文章へのpath
        labels: i番目の形態素から係る係り受けのlabelがlabels[i]に入っている。
        keiDeps: i番目の形態素の係る形態素をkeiDeps[i]としたリスト。ROOTは形態素の個数+1
    '''
    def __init__(self, df, vocab_path, max_len=320, BPE=False):
        self.tokenizer = BertTokenizer(vocab_path, do_lower_case=False,
                                       tokenize_chinese_chars=False)
        # BERT tokenizerでtokenizeする
        df['tokenized_text'] = df['keitaiso'].map(
            lambda x: self.tokenizer.tokenize(" ".join(["[CLS]"] + x +
                                                       ["[SEP]"])))

        # [[値、行のindex、列のindex], ...]形式のmapping_matrixを作る
        if (BPE is True):
            mapping_lists_3 = make_untokenize_mapping_lists(df['keitaiso'],
                                                          df['tokenized_text'])

        # unused1をrootとして使う
        df['tokenized_text'] = df['tokenized_text'].map(
         lambda x:  x+["[unused1]"])

        print("###### Example ######")
        print("tokenized text: ", df['tokenized_text'].to_numpy()[0])

        # Shorten to max length (Bert has a limit of 512)
        df.loc[:, 'tokenized_text'] = df.tokenized_text.str[:max_len]

        df['indexed_tokens'] = df.tokenized_text.map(
            self.tokenizer.convert_tokens_to_ids
        )

        print("converted tokens: ", df['indexed_tokens'].to_numpy()[0])

        sequences = df.indexed_tokens.tolist()
        max_sequence_length = max(len(x) for x in sequences)

        # BERTの入力を作る
        self.inputs_lst, self.masks, self.segments = [], [], []
        for sequence in sequences:
            self.inputs_lst.append(
                sequence+(max_sequence_length-len(sequence))*[0])
            self.masks.append(
                len(sequence)*[1]+(max_sequence_length-len(sequence))*[0])
            self.segments.append(max_sequence_length*[0])

        self.paths = df['path'].to_numpy()
        self.keiDeps = df['dep'].to_numpy()
        self.k2b = df['k2b'].to_numpy()
        self.seq_len = []
        for i in range(len(self.paths)):
            self.k2b[i] = self.k2b[i] + \
                          (max_sequence_length-len(self.k2b[i]))*[-1]
            self.seq_len.append(len(self.keiDeps[i]))
            for j in range(len(self.keiDeps[i])):
                if self.keiDeps[i][j] == 'NULL':
                    self.keiDeps[i][j] = len(self.keiDeps[i])
            self.keiDeps[i] = \
                self.keiDeps[i]+(max_sequence_length-len(self.keiDeps[i]))*[0]
            self.k2b[i]+(max_sequence_length-len(self.k2b[i]))*[-1]

        self.mapping_lists = [None]*len(self.inputs_lst)
        if (BPE is True):
            tokens_processed = df['tokenized_text'].tolist()
            # list形式に変換 [0,1,2,3,3,4,5,6,-1,-1,...]
            max_token_length = max(len(x) for x in tokens_processed)
            self.mapping_lists = []
            for i in range(len(tokens_processed)):
                mapping_list = np.ones(max_token_length)*(-1)
                mapping_list[0] = 0  # CLS
                for j in range(len(mapping_lists_3[i])):
                    mapping_list[j+1] = mapping_lists_3[i][j][1]+1
                mapping_list[len(mapping_lists_3[i])+1] = \
                    mapping_list[len(mapping_lists_3[i])] + 1  # SEP
                self.mapping_lists.append(mapping_list)

    def __getitem__(self, i):
        return self.inputs_lst[i], self.masks[i], self.segments[i], \
               self.paths[i], self.keiDeps[i], \
               self.mapping_lists[i], self.seq_len[i], self.k2b[i]

    def __len__(self):
        return len(self.inputs_lst)


def DepCollate(batch):
    '''
    inputs_lst: 形態素分割された文章をvocabにより数値に変換したリスト
    masks: ↑のlistにおいて、要素が入っている位置を1、要素がない場所を0としたmask
    segment:
    keiDep_matrixs: keiDepsのmatrix表記
    mapping_matrixs: BERTの入力からタスクの単位に直すためのmatrix
    kmasks: 形態素のs_jiをマスクするためのもの, -INFで埋めている
    paths: この文章へのpath
    labels: i番目の形態素から係る係り受けのlabelがlabels[i]に入っている。
    keiDeps: i番目の形態素の係る形態素をkeiDeps[i]としたリスト。ROOTは形態素の個数+1
    '''
    inputs = torch.LongTensor([item[0] for item in batch])
    mask = torch.LongTensor([item[1] for item in batch])
    segment = torch.LongTensor([item[2] for item in batch])
    path = [item[3] for item in batch]
    keiDeps = torch.LongTensor([item[4] for item in batch])
    mapping_lists = [item[5] for item in batch]
    seq_len = torch.LongTensor([item[6] for item in batch])
    k2b = torch.LongTensor([item[7] for item in batch])

    # keiDepを隣接行列になおす+kmasksの作成
    kmasks = []
    keiDep_matrixs = np.zeros(
        (keiDeps.shape[0], keiDeps.shape[1], keiDeps.shape[1]))
    for i in range(keiDeps.shape[0]):
        kmasks.append(
            [0]*(int(seq_len[i]+1))+[-INF]*int(keiDeps.shape[1]-seq_len[i]-1))
        for j in range(seq_len[i]):
            keiDep_matrixs[i, j, keiDeps[i][j]] = 1
    keiDep_matrixs = torch.LongTensor(keiDep_matrixs)
    kmasks = torch.LongTensor(kmasks)
    kmasks = kmasks.view(kmasks.shape[0], kmasks.shape[1], 1) \
             + kmasks.view(kmasks.shape[0], 1, kmasks.shape[1])

    # BPEなしのとき
    if mapping_lists[0] is None:
        mapping_matrixs = np.zeros(
            (keiDeps.shape[0], keiDeps.shape[1], keiDeps.shape[1]))
        for i in range(keiDeps.shape[0]):
            for j in range(seq_len[i]+1):
                mapping_matrixs[i, j, j+1] = 1
        mapping_matrixs = torch.Tensor(mapping_matrixs)

    # BPEありのとき
    if mapping_lists[0] is not None:
        mapping_matrixs = np.zeros((
            keiDeps.shape[0], keiDeps.shape[1], len(mapping_lists[0])))
        for i in range(keiDeps.shape[0]):
            for j in range(1, len(mapping_lists[i])):
                if (mapping_lists[i][j] < 0):
                    mapping_matrixs[i, int(mapping_lists[i][j-1]),
                        batch[i][0].index(5)] = 1
                    break
                mapping_matrixs[i, int(mapping_lists[i][j]-1), j] = 1
            # 重み平均にする
            for j in range(keiDeps.shape[1]):
                if np.sum(mapping_matrixs[i, j, :]) == 0:  # 終わり
                    break
                mapping_matrixs[i, j, :] /= np.sum(mapping_matrixs[i, j, :])
        mapping_matrixs = torch.Tensor(mapping_matrixs)

    # jumanをbunsetsuになおす行列
    k2b_matrixs = np.zeros((
        keiDeps.shape[0], keiDeps.shape[1], keiDeps.shape[1]))
    for i in range(keiDeps.shape[0]):
        length = len(k2b[i])
        for j in range(len(k2b[i])):
            if k2b[i][j] < 0:
                length = j
                break
            k2b_matrixs[i, k2b[i][j], j] = 1
        for j in range(max(k2b[i])+1):
            k2b_matrixs[i, j, :] /= np.sum(k2b_matrixs[i, j, :])
        k2b_matrixs[i, max(k2b[i])+1, length+1] = 1
    k2b_matrixs = torch.Tensor(k2b_matrixs)

    mapping_matrixs = torch.matmul(k2b_matrixs, mapping_matrixs)

    inputs, mask, segment, keiDep_matrixs, mapping_matrixs, \
    kmasks, keiDeps\
        = map(
            lambda x: x.to(DEVICE),
            (inputs, mask, segment, keiDep_matrixs, mapping_matrixs,
            kmasks, keiDeps),
         )

    return inputs, mask, segment, keiDep_matrixs, mapping_matrixs, \
           kmasks, path, keiDeps
