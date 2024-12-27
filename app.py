import streamlit as st 
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate
import shap
import matplotlib.pyplot as plt

# 载入数据
data1 = pd.read_csv("起跑.csv", encoding='gbk')
data1.dropna(inplace=True)
data1.columns = ['Name', 'Achilles tendon stress', 'Ankle plantar/dorsiflexion angle', 
                 'Ankle in/eversion angle', 'Ankle plantar/dorsiflexion moment', 'Ankle in/eversion moment', 
                 'Ankle power', 'A/P GRF', 'Hip ad/abduction moment', 'Hip in/external rotation moment', 
                 'Hip power', 'Knee ad/abduction angle', 'Knee ad/abduction moment', 
                 'Knee in/external rotation moment', 'Ipsi/contralateral pelvic lean', 
                 'EMG activation for gastrocnemius', 'EMG activation for soleus']

X = data1[['Ankle plantar/dorsiflexion angle', 'Ankle in/eversion angle', 'Ankle plantar/dorsiflexion moment', 
           'Ankle in/eversion moment', 'Ankle power', 'A/P GRF', 'Hip ad/abduction moment', 
           'Hip in/external rotation moment', 'Hip power', 'Knee ad/abduction angle', 
           'Knee ad/abduction moment', 'Knee in/external rotation moment', 
           'Ipsi/contralateral pelvic lean', 'EMG activation for gastrocnemius', 
           'EMG activation for soleus']]
y = data1[['Achilles tendon stress']]

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost模型
model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=1, min_child_weight=5,
                         learning_rate=0.03, n_estimators=2000, subsample=1, max_depth=2)
model.fit(X_train, y_train)

# 使用KFold交叉验证
cv = KFold(n_splits=5, shuffle=True, random_state=120)
result = cross_validate(model, X, y, cv=cv, return_train_score=True, verbose=True)

# 计算平均得分
test_mean = result['test_score'].mean()
train_mean = result['train_score'].mean()

# 创建Streamlit界面
st.title("Achilles Tendon Stress Prediction")

# 显示交叉验证结果
st.write(f"测试集平均得分: {test_mean:.4f}")
st.write(f"训练集平均得分: {train_mean:.4f}")

# 显示SHAP解释图
st.subheader("SHAP Explanation")

# Train SHAP模型
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# SHAP summary plot
st.pyplot(shap.summary_plot(shap_values, X))

# 在侧边栏中添加用户输入
st.sidebar.header("Input Parameters")

# 收集用户输入的15个特征
ankle_angle = st.sidebar.slider("Ankle plantar/dorsiflexion angle", min_value=-30, max_value=30, value=0)
ankle_inversion = st.sidebar.slider("Ankle in/eversion angle", min_value=-30, max_value=30, value=0)
ankle_moment = st.sidebar.slider("Ankle plantar/dorsiflexion moment", min_value=-50, max_value=50, value=0)
ankle_inversion_moment = st.sidebar.slider("Ankle in/eversion moment", min_value=-50, max_value=50, value=0)
ankle_power = st.sidebar.slider("Ankle power", min_value=0, max_value=100, value=0)
ap_grf = st.sidebar.slider("A/P GRF", min_value=0, max_value=200, value=0)
hip_ad_abduction_moment = st.sidebar.slider("Hip ad/abduction moment", min_value=-50, max_value=50, value=0)
hip_in_rotation_moment = st.sidebar.slider("Hip in/external rotation moment", min_value=-50, max_value=50, value=0)
hip_power = st.sidebar.slider("Hip power", min_value=0, max_value=100, value=0)
knee_ad_abduction_angle = st.sidebar.slider("Knee ad/abduction angle", min_value=-30, max_value=30, value=0)
knee_ad_abduction_moment = st.sidebar.slider("Knee ad/abduction moment", min_value=-50, max_value=50, value=0)
knee_in_rotation_moment = st.sidebar.slider("Knee in/external rotation moment", min_value=-50, max_value=50, value=0)
pelvic_lean = st.sidebar.slider("Ipsi/contralateral pelvic lean", min_value=-30, max_value=30, value=0)
gastrocnemius_emg = st.sidebar.slider("EMG activation for gastrocnemius", min_value=0, max_value=100, value=0)
soleus_emg = st.sidebar.slider("EMG activation for soleus", min_value=0, max_value=100, value=0)

# 将用户输入的数据整理成模型需要的输入形式
user_input = np.array([[ankle_angle, ankle_inversion, ankle_moment, ankle_inversion_moment, ankle_power, ap_grf,
                        hip_ad_abduction_moment, hip_in_rotation_moment, hip_power, knee_ad_abduction_angle,
                        knee_ad_abduction_moment, knee_in_rotation_moment, pelvic_lean, gastrocnemius_emg, soleus_emg]])

# 预测
predicted_stress = model.predict(user_input)
st.write(f"预测的跟腱应力: {predicted_stress[0]:.2f}")