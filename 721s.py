# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:23:17 2024

@author: gaoli
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle

# Load the model
model = joblib.load('xgboost_reduced_features_model.pkl')

# Load the scaler
scaler = joblib.load('scaler_reduced_features.pkl')

# Load selected feature names
with open('selected_feature_names.pkl', 'rb') as f:
    selected_feature_names = pickle.load(f)

# 不需要手动定义 feature_names，直接使用 selected_feature_names
feature_names = selected_feature_names

# 确保特征名称与模型预期的输入匹配
if list(selected_feature_names) != feature_names:
    st.error("Feature names do not match the model's expected input.")
    st.stop()

# Streamlit user interface
st.title("Type 2 Diabetes Risk Predictor")

# GLU: numerical input
GLU = st.number_input("Glucose (GLU) (mmol/L):", min_value=0.0, max_value=50.0, value=5.0, step=0.1)

# LDL: numerical input, with value range starting from 0
LDL = st.number_input("Low-density Lipoprotein Cholesterol (LDL) (mmol/L):", min_value=0.0, max_value=100.0, value=3.0, step=0.1)

# Tyg: numerical input
Tyg = st.number_input("TyG Index:", min_value=0.0, max_value=50.0, value=8.0, step=0.01)

# weight: numerical input
weight = st.number_input("Weight (kg):", min_value=0.0, max_value=300.0, value=70.0, step=0.1)

# age: numerical input
age = st.number_input("Age (years):", min_value=0, max_value=120, value=50, step=1)

# WBC: numerical input
WBC = st.number_input("White Blood Cells (WBC) (10^9/L):", min_value=0.0, max_value=100.0, value=7.0, step=0.1)

# CRP: numerical input
CRP = st.number_input("C-reactive Protein (CRP) (mg/L):", min_value=0.0, max_value=200.0, value=1.0, step=0.1)

# CAR: numerical input
CAR = st.number_input("CRP to Albumin Ratio (CAR):", min_value=0.0, max_value=50.0, value=0.5, step=0.01)

# creatinine: numerical input
creatinine = st.number_input("Creatinine (Scr) (μmol/L):", min_value=0.0, max_value=500.0, value=5.0, step=0.1)


# Collect feature values in the correct order
feature_values = [WBC, age, GLU, creatinine, CAR, weight, LDL, Tyg, CRP]

# Convert to numpy array and reshape
features = np.array([feature_values])

# Scale features using the loaded scaler
features_scaled = scaler.transform(features)

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features_scaled)[0]
    predicted_proba = model.predict_proba(features_scaled)[0]

    # Display prediction results
    if predicted_class == 1:
        st.write("**Prediction:** High Risk of Diabetes")
    else:
        st.write("**Prediction:** Low Risk of Diabetes")

    st.write(f"**Probability of Diabetes:** {predicted_proba[1]*100:.2f}%")
    st.write(f"**Probability of Not Having Diabetes:** {predicted_proba[0]*100:.2f}%")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of diabetes. "
            f"The model predicts that your probability of having diabetes is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "We recommend that you consult an endocrinologist as soon as possible for further evaluation "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of diabetes. "
            f"The model predicts that your probability of not having diabetes is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "We recommend regular check-ups to monitor your health "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

    # 定义一个函数来在 Streamlit 中显示 SHAP 图
def st_shap(plot, height=None):
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# 计算 SHAP 值并显示力图
# 创建单个输入的 DataFrame
input_df = pd.DataFrame([feature_values], columns=feature_names)

# 创建 SHAP explainer
explainer = shap.TreeExplainer(model)

# 计算 SHAP 值
shap_values = explainer.shap_values(features_scaled)

# 检查 explainer.expected_value 和 shap_values 的类型和形状
st.write(f"explainer.expected_value: {explainer.expected_value}")
st.write(f"shap_values.shape: {shap_values.shape}")

# 对于二分类问题，shap_values 通常是一个二维数组，形状为 (n_samples, n_features)
# explainer.expected_value 是一个标量值

# 显示第一个（也是唯一一个）样本的力图
shap.initjs()
st.write("**SHAP Force Plot:**")
shap_force_plot = shap.force_plot(explainer.expected_value, shap_values[0], input_df)
st_shap(shap_force_plot)





