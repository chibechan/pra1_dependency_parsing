# 係り受け解析の実装
## 実装の説明
model:   
係受け解析で使うモデル  
[この論文](https://www.anlp.jp/proceedings/annual_meeting/2019/pdf_dir/F2-4.pdf)を実装している  

utils:  
データの前処理  

mapping.py: 文節単位と形態素単位を結びつけるアルゴリズム  
preprocess_dep.py: 係り受け解析ようにDatasetとDataLoaderを用意する  
preprocess_KWDLC: KWDLCのデータを整形  

## デモ
[このリンク](https://colab.research.google.com/drive/1xeRZncvw1I-1LfeH_9rqTM6wnSw7Sy2O?usp=sharing)でKWDLCの実験を行える

## KWDLC以外の係り受け解析を実装する方法
preprocess_KWDLC.pyのmake_df_KWDLCを参考にして以下のcolumnsをを持つDataFrameを作る．  

path: 文章へのpath（基本的になんでもOK)  
keitaiso: 文章の形態素分割を保持したリスト  
bunsetu: 文章の文節分割を保持したリスト  
dep: 文節単位の係り受け  
k2b: それぞれの形態素がどこの基本句に対応しているかを保存したリスト  

これさえ作れば，後はKWDLCと同じように係受け解析のtrainingとtestを行える．
