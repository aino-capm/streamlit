#ライブラリの読み込み
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os

#タイトル
st.markdown("## 類似文書検索")
st.caption("Doc2Vecで有価証券報告書のテキスト文書をベクトル化。選択企業と類似した文書を作成している企業を検索します")

pages = dict(
    page1="TF-IDF",
    page2="類似文書検索",
)

page_id = st.sidebar.selectbox(
    "menu",
    
    ["page1", "page2"],
    format_func=lambda page_id: pages[page_id],
    key="page-select",
)

#ファイルアップロード
FILE_PATH ="files"
csv_file = st.sidebar.file_uploader('csvファイルをアップロードしてください.', type="csv")
if csv_file:
  st.sidebar.markdown(f'{csv_file.name} をアップロードしました.')
  file_path = os.path.join(FILE_PATH, csv_file.name)
  with open(file_path, 'wb') as f:
    f.write(csv_file.read())
  df = pd.read_csv(file_path,index_col=0)
  
  corp = df["会社名"]
  x = st.selectbox("企業を選択してください",corp)
  index = df.loc[df["会社名"]==x].index[0]
  
  st.markdown("### 分析テキストの選択")
    
  text = st.radio("選択してください",('経営方針','事業等のリスク'))
  slider = st.slider("表示文字数",min_value=300,max_value=2500)
  st.write(df.iloc[index][text][:slider])
    

  st.markdown("### 文章ベクトルの計算&類似文書の検索")
  st.caption("ベクトルのサイズ・エポック数を入力してください")
  #データフレームを表示
  # #実際は計算させる。暫定的に計算済みのファイルを使用する
  df_vec = pd.read_csv("files/文書ベクトル計算用2103.csv",index_col=0)
  sentences,labels = [],[]
  for i in range(len(df_vec)):
    text = df_vec.iloc[i,1]
    text_list = text.split(' ')
    sentences.append(text_list)
    labels.append(df_vec.iloc[i,2])
  documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
  
  vector_size = st.number_input("ベクトルのサイズ",min_value=100,max_value=300,value=300,step=50)
  epochs = st.number_input("エポック数",min_value=10,max_value=20,value=20,step=5)

  execute1 = st.button("計算実行")
    #実行ボタンを押したら下記が進む
  if execute1:
    model = Doc2Vec(documents, vector_size=vector_size,  window=7, min_count=1, workers=4, epochs=epochs)
    
    st.markdown("#### 検索結果")
    result = pd.DataFrame(model.docvecs.most_similar(index),columns=["index","類似度"])
    result["会社名"] = result["index"].apply(lambda x : df_vec.iloc[x,0])
    st.write(result)


