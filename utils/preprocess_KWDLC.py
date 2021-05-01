import os
import pandas as pd


def make_df_KWDLC(data_path="./data/KWDLC-1.0/dat/rel/"):
    '''
    input:
        KWDLCの係り受け解析のデータへのpath
    output:
        以下の要素を持っているdataframeを返す
            path: 文章へのpath（基本的になんでもOK)
            keitaiso: 文章の形態素分割を保持したリスト
            bunsetu: 文章の文節分割を保持したリスト
            dep: 文節単位の係り受け
            k2b: それぞれの形態素がどこの基本句に対応しているかを保存したリスト
    '''
    k = ''  # 基本句の1単位
    b = ''  # 文節の1単位
    kcount = -1  # 基本句の番号
    bcount = -1  # 文節の番号
    kei = []  # 形態素
    kih= []  # 基本句
    bun = []  # 文節
    kihDep = []  # 基本句のDependency
    bunDep = []  # 文節のDependency
    k2k = []  # 形態素から基本句
    k2b = []  # 形態素から文節
    data_files = []
    keis = []
    kihs = []
    buns = []
    kihDeps = []
    bunDeps = []
    k2ks = []
    k2bs = []

    for dir_num in sorted(os.listdir(data_path)):
        for file_num in sorted(os.listdir(data_path+dir_num)):
            # 開くファイルの名前↓
            data_file = data_path + dir_num + '/' + file_num
            with open(data_file) as f:
                sen = f.readline()
                while(sen):
                    sen = sen.strip()
                    if (sen=='EOS'): # 一文の終わり
                        kihDep[kihDep.index(-1)] = max(k2k)+1
                        bunDep[bunDep.index(-1)] = max(k2b)+1
                        kih.append(k)
                        bun.append(b)
                        data_files.append(data_file)
                        keis.append(kei)
                        kihs.append(kih)
                        buns.append(bun)
                        kihDeps.append(kihDep)
                        bunDeps.append(bunDep)
                        k2ks.append(k2k)
                        k2bs.append(k2b)
                        k = ''
                        b = ''
                        kcount = -1
                        bcount = -1
                        kei = []
                        kih = []
                        bun = []
                        kihDep = []
                        bunDep = []
                        k2k = []
                        k2b = []
                    elif (sen[0]=='*'): # 文節単位
                        if (b!= ''):
                            bun.append(b)
                            b = ''
                        bunDep.append(int(sen.split()[2][:-1]))
                        bcount += 1
                    elif (sen[0]=='+'): # 基本句単位
                        if (k!=''):
                            kih.append(k)
                            k = ''
                        kihDep.append(int(sen.split()[2][:-1]))
                        kcount += 1
                    else: # 単語
                        if (sen.split()[0] != '#'):
                            k += sen.split()[0]
                            b += sen.split()[0]
                            kei.append(sen.split()[0])
                            k2k.append(kcount)
                            k2b.append(bcount)
                    sen = f.readline()
    df = pd.DataFrame()
    df['path'] = data_files
    df['keitaiso'] = keis
    df['bunsetu'] = buns
    df['dep'] = bunDeps
    df['k2b'] = k2bs
    return df
