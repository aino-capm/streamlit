#ライブラリの読み込み
import streamlit as st
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import pickle

#タイトル
st.markdown("## 類似文書検索")
st.caption("Doc2Vecで有価証券報告書のテキスト文書をベクトル化。選択企業と類似した文書を作成している企業を検索します")
st.markdown("***")



df = pd.read_csv("files/2203有報セット.csv",index_col=0).reset_index()  
corp = df["会社名"]

@st.cache
def doc2vec(sentences,vector_size,epochs):
  documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
  model = Doc2Vec(documents, vector_size=vector_size,  window=7, min_count=1, workers=4, epochs=epochs)
  
  return model

#ラジオボタンで、テキスト文書を選択する
state = st.radio("選択してください",("経営方針","事業等のリスク"))
if state == "経営方針":
  file_path = "models/sentences_keiei_2203.bin"
else:
  file_path = "models/sentences_risk_2203.bin"
#選択したテキスト文書を呼び出す
with open(file_path,"rb") as p:
  sentences = pickle.load(p)

#計算実行の前提の実装
with st.form("form"):
  x = st.selectbox("企業を選択してください",corp)
  index = df.loc[df["会社名"]==x].index[0]
  st.caption("ベクトルのサイズ・エポック数を入力してください")
  vector_size = st.number_input("ベクトルのサイズ",min_value=100,max_value=300,value=300,step=50)
  epochs = st.number_input("エポック数",min_value=10,max_value=20,value=20,step=5)
  submittted = st.form_submit_button("計算実行")
  if submittted:
    model = doc2vec(sentences,vector_size,epochs)
  
    st.markdown("#### 検索結果")
    result = pd.DataFrame(model.docvecs.most_similar(index),columns=["index","類似度"])
    result["会社名"] = result["index"].apply(lambda x : df.iloc[x,4])
    st.write(result)
