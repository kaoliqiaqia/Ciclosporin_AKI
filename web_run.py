import streamlit as st
import pandas as pd
import joblib

# 页面设置
st.set_page_config(page_title="AKI风险预测", layout="centered")
st.title("急性肾损伤（AKI）风险评估工具")
st.markdown("请选择患者组别并填写以下临床信息")


adult_model_path = "logistic_model_5features.pkl"
adult_bundle_path = "logistic_preprocess_bundle_5features.pkl"
child_model_path = "randomforest_model_top10_child.pkl"
child_bundle_path = "randomforest_preprocess_bundle_top10_child.pkl"

# 加载模型和预处理对象
adult_model = joblib.load(adult_model_path)
adult_bundle = joblib.load(adult_bundle_path)
adult_imputer = adult_bundle["imputer"]
adult_encoder = adult_bundle["encoder"]
adult_scaler = adult_bundle["scaler"]
adult_cont_features = adult_bundle["continuous_features"]
adult_cat_features = adult_bundle["categorical_features"]

child_model = joblib.load(child_model_path)
child_bundle = joblib.load(child_bundle_path)
child_imputer = child_bundle["imputer"]
child_encoder = child_bundle["encoder"]
child_scaler = child_bundle["scaler"]
child_cont_features = child_bundle["continuous_features"]
child_cat_features = child_bundle["categorical_features"]

# 用户选择组别
group = st.radio("请选择患者组别", ("成年组 (≥18岁)", "未成年组 (<18岁)"))

#dynamic generation of input form
if group == "成年组 (≥18岁)":
    st.subheader("成年组临床指标")
    input_data = {}

    for feat in adult_cont_features:
        input_data[feat] = st.number_input(f"{feat} (mg)", min_value=0.0, value=50.0, step=5.0)

    for feat in adult_cat_features:

        display_name = feat
        input_data[feat] = st.selectbox(display_name, [0, 1], format_func=lambda x: "是" if x==1 else "否")
else:
    st.subheader("未成年组临床指标")
    input_data = {}

    for feat in child_cont_features:
        if feat == "alt":
            input_data[feat] = st.number_input("ALT (U/L)", min_value=0.0, value=30.0, step=5.0)
        elif feat == "报告值":
            input_data[feat] = st.number_input("环孢素报告值 (ng/mL)", min_value=0.0, value=100.0, step=10.0)
        else:
            input_data[feat] = st.number_input(feat, min_value=0.0, value=0.0, step=1.0)

    cat_name_map = {
        "凝血功能异常": "凝血功能异常",
        "肝损害": "肝损害",
        "免疫球蛋白缺乏": "免疫球蛋白缺乏",
        "ACEI": "ACEI 使用",
        "念珠菌感染": "念珠菌感染",
        "肺炎": "肺炎",
        "抗胆碱药": "抗胆碱药使用",
        "生长抑素及其类似物": "生长抑素使用"
    }
    for feat in child_cat_features:
        display = cat_name_map.get(feat, feat)
        input_data[feat] = st.selectbox(display, [0, 1], format_func=lambda x: "是" if x==1 else "否")

if st.button("预测 AKI 风险"):

    df_input = pd.DataFrame([input_data])
    
    if group == "成年组 (≥18岁)":
        imputer = adult_imputer
        encoder = adult_encoder
        scaler = adult_scaler
        cont_features = adult_cont_features
        cat_features = adult_cat_features
        model = adult_model
    else:
        imputer = child_imputer
        encoder = child_encoder
        scaler = child_scaler
        cont_features = child_cont_features
        cat_features = child_cat_features
        model = child_model
    
  
    X_imp = imputer.transform(df_input)
    X_imp = pd.DataFrame(X_imp, columns=df_input.columns)
    

    if cat_features:
        cat_df = X_imp[cat_features].copy()
        cat_encoded = encoder.transform(cat_df)

        cat_columns = []
        for i, col in enumerate(cat_features):
            cats = encoder.categories_[i]
            cat_columns.extend([f"{col}_{cat}" for cat in cats[1:]])
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=cat_columns)
        X_imp = X_imp.drop(columns=cat_features)
        X_imp = pd.concat([X_imp, cat_encoded_df], axis=1)
    

    if cont_features:
        X_imp[cont_features] = scaler.transform(X_imp[cont_features])

    prob = model.predict_proba(X_imp)[0, 1]
    pred = 1 if prob >= 0.5 else 0
    

    st.subheader("预测结果")
    if pred == 1:
        st.error(f"⚠️ 高风险：AKI 发生概率为 {prob:.2%}")
        st.markdown("建议立即进行临床评估和进一步检查。")
    else:
        st.success(f"✅ 低风险：AKI 发生概率为 {prob:.2%}")
        st.markdown("继续监测，注意高危因素。")
    
    if group == "成年组 (≥18岁)":
        st.caption("模型基于成年组数据构建，AUC=0.877 (95% CI 0.764–0.966)，阈值0.5。")
    else:
        st.caption("模型基于未成年组数据构建，AUC=0.788 (95% CI 未提供)，阈值0.5。")
    st.caption("预测结果仅供临床参考，不能替代医生诊断。")


