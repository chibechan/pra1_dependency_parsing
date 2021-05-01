from transformers import BertModel
from torch import nn
from model.dep_module import DepModule


class BERT_Parsing(nn.Module):
    def __init__(self, model_path, dep_dim):
        super(BERT_Parsing, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        # dependency parsing
        self.dep_parsing = DepModule(hidden_size, dep_dim)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                s_mask=None, dep=None, mapping_matrix=None, dep_label=None,
                dep_list=None, train=True, ud=True):
        '''
        Dependency:
            input:
                s_mask: Dependencyに関係ないところをmaskする隣接行列
                dep: Dependencyを隣接行列形式で表したもの, predictionの時はNone
                mapping_matrix: 形態素を基本句or文節に変換するためのmatrix
                dep_label: それぞれの係り受けのラベル, predictionのときはNone
                       (ただし、dep_labelのpredictionも行うときはTrue)
                dep_list: Dependencyをリスト形式で表したもの
                　　　　　　predictionのときはNone
                train: trainかpredictionか
                ud: universal dependencyのとき（つまりdep_labelあり)のときはtrue
            output:
                dep_loss: depndencyのheadを予測する問題とdep_labelを予測する問題のlossを足し合わせた
                dep_prediction: headのprediction。行列形式で答えを返す
                dep_label_prediction: dep_labelのprediction。list形式で答えを解す。
        '''

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        # ['[CLS]', 'コイン', 'トス', 'を', '３', '回', '行う', '。', '[SEP]']
        pooled_output = outputs[0]  # batch * Seq_len * hidden

        # dependencpy parsingを行うとき
        if s_mask is not None:
            dep_loss, dep_prediction, dep_label_prediction \
                = self.dep_parsing(pooled_output, s_mask, dep, mapping_matrix,
                                   dep_label, dep_list, train, ud)
            return dep_loss, dep_prediction, dep_label_prediction
