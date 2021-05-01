import re


def make_untokenize_mapping_lists(origin_lists, tokenized_lists):
    """
    BERTによってトークナイズされる前後のリストを受け取り、トークナイズされた形式から本来の形式に戻す疎行列の値を返す。
    input:
        origin_lists: トークナイズされる以前の形式の文章のリスト。文章数N×単語数woの大きさ
        tokenized_lists: トークナイズされた形式の文章のリスト。文章数N×単語数wbの大きさ
    output:
        mapping_list:
            トークナイズされた形式から本来の形式に戻すための行列を作るためのリスト
            [値, 行のindex, 列のindex]を要素とする[文章数, 単語数]の大きさのリスト
    """
    assert len(origin_lists) == len(tokenized_lists)

    mapping_lists = []
    for origin_sentence, tokenized_sentence in zip(origin_lists, tokenized_lists):
        mapping_list = []
        j = 0
        # 元々の単語にスペースが含まれていればBERTはそこで分割しているのでスペースを消去する
        origin_sentence = [re.sub(r"[\u3000 \t]", "", word) for word in origin_sentence]
        tokenized_sentence = replace_unk(origin_sentence, tokenized_sentence)
        for i, origin_word in enumerate(origin_sentence):
            while True:
                tokenized_word = tokenized_sentence[j]
                if tokenized_word == '[CLS]' or tokenized_word == '[SEP]':
                    j += 1
                    continue
                elif origin_word == tokenized_word:
                    mapping_list.append([1, i, j])
                    j += 1
                    break
                elif origin_word.startswith(tokenized_word):
                    mapping_num = 1
                    tmp_mapping_list = [j]
                    j += 1
                    while True:
                        try:
                            next_tokenized_word = tokenized_sentence[j]
                        except IndexError:
                            err_msg = "original sentence: {} \n".format(
                                origin_sentence)
                            err_msg += "tokenized sentence: {}".format(
                                tokenized_sentence)
                            err_msg += "original word: {}".format(origin_word)
                            err_msg += "tokenized word: {}".format(tokenized_word)
                            raise ValueError(err_msg)
                        if next_tokenized_word[:2] == '##':
                            next_tokenized_word = next_tokenized_word[2:]
                        tokenized_word += next_tokenized_word
                        mapping_num += 1
                        tmp_mapping_list.append(j)
                        j += 1
                        if origin_word == tokenized_word:
                            for val in tmp_mapping_list:
                                mapping_list.append([1 / mapping_num, i, val])
                            break
                    break
                else:
                    err_msg = "original sentence: {} \n".format(origin_sentence)
                    err_msg += "tokenized sentence: {}".format(tokenized_sentence)
                    err_msg += "original word: {}".format(origin_word)
                    err_msg += "tokenized word: {}".format(tokenized_word)
                    raise ValueError(err_msg)
        mapping_lists.append(mapping_list)
    return mapping_lists


def replace_unk(origin_sentence, tokenized_sentence):
    """
    tokenized_sentenceの[UNK]に元の文章の単語を埋め込む
    origin_sentence
        ['ヾ', '（〃＾∇＾）', 'ノ']
    tokenized_sentence
        ['[CLS]', '[UNK]', '（', '[UNK]', '[UNK]', '）', 'ノ', '[SEP]']
    のように一意に[UNK]の中身が決定できない場合は適当な中身を入れる
    input:
        origin_sentence: トークナイズされる以前の形式の文章の単語のリスト。単語数woの大きさ
        tokenized_sentence: トークナイズされた形式の文章の単語のリスト。単語数wbの大きさ
    output:
        tokenized_sentence:
            [UNK]に元の文章の単語を代入されたトークナイズされた形式の文章の単語のリスト。単語数wbの大きさ
    """
    UNK = '[UNK]'
    if UNK not in tokenized_sentence:
        return tokenized_sentence
    content_list = []
    impossible_content_list = []
    unk_num = 0
    unk_idx_list = []
    i = 0
    while i < len(tokenized_sentence):
        word = tokenized_sentence[i]
        if word == UNK:
            tmp_unk_num = 1
            unk_idx_list.append(i)
            next_word = tokenized_sentence[i + 1]
            while next_word == UNK and i + 1 < len(tokenized_sentence):
                i += 1
                tmp_unk_num += 1
                unk_idx_list.append(i)
                next_word = tokenized_sentence[i + 1]
            unk_num += tmp_unk_num
            content_list.append('X?' + 'X'.join(['([^X]+?)'] * tmp_unk_num))
            impossible_content_list.append('X?' + 'X?'.join(['([^X]+?)'] * tmp_unk_num))
        elif word == '[CLS]' or word == '[SEP]':
            i += 1
            continue
        else:
            if word[:2] == '##':
                word = word[2:]
            # "("などの特殊文字をエスケープするために[]で囲む
            content_list.append('X?' + '[' + ']['.join(list(word)) + ']')
            impossible_content_list.append('X?' + '[' + ']['.join(list(word)) + ']')
        i += 1
    pattern = re.compile(''.join(content_list + [r'\Z']))
    impossible_pattern = re.compile(''.join(impossible_content_list + [r'\Z']))
    # [UNK]が連続する部分の区切りを判別するために単語をXで区切る
    result = pattern.match('X' + 'X'.join(origin_sentence))
    impossible_result = impossible_pattern.match('X' + 'X'.join(origin_sentence))
    for i in range(unk_num):
        if result is not None:
            unk_string = result.group(i + 1)
        else:
            unk_string = impossible_result.group(i + 1)
        tokenized_sentence[unk_idx_list[i]] = unk_string
    return tokenized_sentence
