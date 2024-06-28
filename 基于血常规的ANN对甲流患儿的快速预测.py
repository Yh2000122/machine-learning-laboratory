
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# 加载模型和归一化器
model = joblib.load('ann_model.pkl')
scaler = joblib.load('scaler.pkl')

# 设置页面标题
st.title('基于血常规的ANN模型对甲流患儿的快速预测')

# 创建输入表单
st.sidebar.header('输入血常规')


def user_input_features():
    features = {}
    # 使用与模型训练时相同的特征名称
    feature_names = ['MO(*10^9/L)', 'PCT', 'PLT(*10^9/L)', 'SAA(mg/L)', 'LYM(*10^9/L)',
                     'PLR', 'CRP(＜9.9mg/L填0，＞9.9mg/L填1)', 'PDW(fl)', 'LYM×PLT', 'NLR',
                     'LY%']
    for column in feature_names:
        features[column] = st.sidebar.number_input(column, value=0.0)
    return pd.DataFrame(features, index=[0])


input_df = user_input_features()

st.subheader('输入值预览')
st.write(input_df)

if st.button('预测'):
    # 归一化输入数据
    input_scaled = scaler.transform(input_df)

    # 进行预测
    prediction = model.predict(input_scaled)
    st.subheader('预测结果')
    st.write(prediction)