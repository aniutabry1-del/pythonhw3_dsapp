import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# НАСТРОЙКИ СТРАНИЦЫ
# ============================================================
st.set_page_config(
    page_title="Income Predictor",
    page_icon="💸",
    layout="centered"
)

# ============================================================
# ЗАГОЛОВОК
# ============================================================
st.title("💸 Your Income Predictor")
st.subheader("Input your data and check if your income will surpass $50K")
st.markdown("---")

# ============================================================
# ЗАГРУЗКА И ОБУЧЕНИЕ МОДЕЛИ (кэшируем для скорости)
# ============================================================
@st.cache_resource
def load_and_train_model():
    """Загружает данные и обучает модель"""
    # Загрузка данных
    url = "https://raw.githubusercontent.com/aniutabry1-del/pythonhw3_dsapp/main/data.adult.csv"
    try:
        df = pd.read_csv(url)
    except:
        # Если не получилось загрузить с GitHub, попробуем локально
        df = pd.read_csv("data.adult.csv")
    
    # Удаляем строки с пропусками (обозначены как '?')
    df_clean = df[~(df == '?').any(axis=1)]
    
    # Целевая переменная
    target = '>50K,<=50K'
    y = (df_clean[target] == '>50K').astype(int)
    X = df_clean.drop(columns=[target])
    
    # Разделяем на числовые и категориальные
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Масштабируем числовые
    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(
        scaler.fit_transform(X[numerical_cols]), 
        columns=numerical_cols, 
        index=X.index
    )
    
    # One-hot encoding для категориальных
    X_cat_encoded = pd.get_dummies(X[categorical_cols], drop_first=True)
    
    # Объединяем
    X_full = pd.concat([X_num_scaled.reset_index(drop=True), X_cat_encoded.reset_index(drop=True)], axis=1)
    
    # Обучаем GradientBoosting
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_full, y.reset_index(drop=True))
    
    return model, scaler, numerical_cols, categorical_cols, X_cat_encoded.columns.tolist()

# Загружаем модель
with st.spinner("Loading model..."):
    model, scaler, numerical_cols, categorical_cols, encoded_columns = load_and_train_model()

# ============================================================
# ОПЦИИ ВВОДА ДАННЫХ
# ============================================================
st.header("📊 Input Data")

input_method = st.radio(
    "Choose input method:",
    ["Manual Input", "Upload CSV File"],
    horizontal=True
)

# ============================================================
# РУЧНОЙ ВВОД ДАННЫХ
# ============================================================
if input_method == "Manual Input":
    st.markdown("#### Enter your information:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=17, max_value=100, value=35)
        
        workclass = st.selectbox("Work Class", [
            "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
            "Local-gov", "State-gov", "Without-pay", "Never-worked"
        ])
        
        education = st.selectbox("Education", [
            "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
            "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
            "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
        ])
        
        education_num = st.slider("Education Years", min_value=1, max_value=16, value=10)
        
        marital_status = st.selectbox("Marital Status", [
            "Married-civ-spouse", "Divorced", "Never-married", "Separated",
            "Widowed", "Married-spouse-absent", "Married-AF-spouse"
        ])
        
        occupation = st.selectbox("Occupation", [
            "Tech-support", "Craft-repair", "Other-service", "Sales",
            "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
            "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
            "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
        ])
    
    with col2:
        relationship = st.selectbox("Relationship", [
            "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
        ])
        
        race = st.selectbox("Race", [
            "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
        ])
        
        sex = st.selectbox("Sex", ["Male", "Female"])
        
        capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
        
        capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0)
        
        hours_per_week = st.slider("Hours per Week", min_value=1, max_value=100, value=40)
        
        fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1500000, value=200000)
    
    # Создаём DataFrame из введённых данных
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'education': [education],
        'education-num': [education_num],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week]
    })
    
    data_ready = True

# ============================================================
# ЗАГРУЗКА CSV ФАЙЛА
# ============================================================
else:
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write("**Preview of uploaded data:**")
        st.dataframe(input_data.head())
        data_ready = True
    else:
        data_ready = False
        st.info("👆 Please upload a CSV file with the required columns")

# ============================================================
# КНОПКА ПРЕДСКАЗАНИЯ
# ============================================================
st.markdown("---")

if data_ready:
    if st.button("🔮 Get Prediction", type="primary", use_container_width=True):
        
        with st.spinner("Analyzing..."):
            # Подготовка данных для предсказания
            try:
                # Масштабируем числовые признаки
                X_num = input_data[numerical_cols]
                X_num_scaled = pd.DataFrame(
                    scaler.transform(X_num),
                    columns=numerical_cols
                )
                
                # One-hot encoding для категориальных
                X_cat = input_data[categorical_cols]
                X_cat_encoded = pd.get_dummies(X_cat, drop_first=True)
                
                # Добавляем отсутствующие столбцы (которые были при обучении)
                for col in encoded_columns:
                    if col not in X_cat_encoded.columns:
                        X_cat_encoded[col] = 0
                
                # Убираем лишние столбцы и сортируем
                X_cat_encoded = X_cat_encoded.reindex(columns=encoded_columns, fill_value=0)
                
                # Объединяем
                X_full = pd.concat([X_num_scaled.reset_index(drop=True), X_cat_encoded.reset_index(drop=True)], axis=1)
                
                # Предсказание
                predictions = model.predict(X_full)
                probabilities = model.predict_proba(X_full)[:, 1]
                
                # ============================================================
                # ВЫВОД РЕЗУЛЬТАТА
                # ============================================================
                st.markdown("---")
                st.header("🎯 Prediction Result")
                
                for i in range(len(predictions)):
                    prob = probabilities[i] * 100
                    
                    if predictions[i] == 1:
                        st.success(f"### ✅ Income > $50K")
                        st.balloons()
                    else:
                        st.warning(f"### ❌ Income ≤ $50K")
                    
                    # Красивый прогресс-бар
                    st.markdown(f"**Probability of earning >$50K: {prob:.1f}%**")
                    st.progress(prob / 100)
                    
                    # График
                    fig, ax = plt.subplots(figsize=(8, 3))
                    colors = ['#ff6b6b' if prob < 50 else '#51cf66']
                    ax.barh(['Your Probability'], [prob], color=colors, height=0.5)
                    ax.axvline(x=50, color='gray', linestyle='--', label='Threshold (50%)')
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('Probability (%)')
                    ax.legend()
                    ax.set_title('Income Prediction Confidence')
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.write("Please check that your data has all required columns.")

# ============================================================
# САЙДБАР
# ============================================================
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This app predicts whether a person's income exceeds $50K based on census data.

**Model:** Gradient Boosting Classifier

**Features used:**
- Age, Education, Work hours
- Occupation, Work class
- Marital status, Relationship
- And more...

**Author:** Брылева  
**Course:** Python for Data Analysis (HSE)
""")

st.sidebar.markdown("---")
st.sidebar.write("📊 Based on UCI Adult Census Dataset")

