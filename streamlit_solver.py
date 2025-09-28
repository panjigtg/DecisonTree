
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Decision Tree Fleksibel", layout="wide")

def load_data(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type == 'csv':
            return pd.read_csv(uploaded_file)
        elif file_type == 'xlsx':
            return pd.read_excel(uploaded_file, engine='openpyxl')
        elif file_type == 'xls':
            return pd.read_excel(uploaded_file)
        elif file_type == 'tsv':
            return pd.read_csv(uploaded_file, sep='\t')
        elif file_type == 'txt':
            content = uploaded_file.read().decode('utf-8')
            delimiter = ',' if ',' in content else ('\t' if '\t' in content else ';')
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=delimiter)
        else:
            raise ValueError("Tipe file tidak didukung.")
    except Exception as e:
        raise ValueError(f"Gagal membaca file: {e}")

def detect_problem_type(y):
    if pd.api.types.is_numeric_dtype(y):
        unique_ratio = y.nunique() / len(y)
        return "regression" if unique_ratio > 0.05 or y.nunique() > 20 else "classification"
    return "classification"

def preprocess_data(X, y, problem_type):
    X = X.fillna(X.mean(numeric_only=True) if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
    for col in X.select_dtypes(include='object'):
        if X[col].nunique() > 100:
            st.warning(f"⚠️ Kolom '{col}' memiliki terlalu banyak kategori ({X[col].nunique()}), akan dihapus.")
            X = X.drop(columns=[col])
    X_processed = pd.get_dummies(X, drop_first=True)
    y_processed = y.copy()
    label_encoder = None
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y_processed = label_encoder.fit_transform(y)
    return X_processed, y_processed, label_encoder

def plot_optimized_tree(model, feature_names, class_names, max_depth_display=3):
    n_features = len(feature_names)
    width = max(12, min(20, n_features * 2))
    height = max(8, min(16, max_depth_display * 3))
    fig, ax = plt.subplots(figsize=(width, height))
    safe_class_names = [str(c) for c in class_names] if class_names is not None else None
    plot_tree(model, feature_names=feature_names, class_names=safe_class_names,
              filled=True, rounded=True, fontsize=max(8, min(12, 100 // max(1, n_features))),
              max_depth=max_depth_display, ax=ax)
    plt.tight_layout()
    return fig

st.title("🌐 Decision Tree Classifier/Regressor - Fleksibel")

uploaded_file = st.file_uploader("📂 Upload file data (.csv, .xlsx, .xls, .txt, .tsv)", 
                                  type=["csv", "xlsx", "xls", "txt", "tsv"])

if uploaded_file:
    try:
        df = load_data(uploaded_file)
        st.success("✅ File berhasil dibaca!")
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

    st.metric("Jumlah Baris", len(df))
    st.metric("Jumlah Kolom", len(df.columns))
    st.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("👀 Preview Data")
    st.dataframe(df.head())

    st.subheader("🎯 Pilih Kolom Target (Y)")
    target_column = st.selectbox("Kolom yang ingin diprediksi:", df.columns)

    if target_column:
        y = df[target_column]
        st.subheader("🔎 Preview Target")
        st.write(y.value_counts())

        problem_type = detect_problem_type(y)
        st.metric("Tipe Problem", problem_type.title())
        st.metric("Unique Values", int(y.nunique()))
        st.metric("Data Type", str(y.dtype))

        if problem_type == "regression":
            if st.checkbox("🔄 Paksa sebagai Classification (binning otomatis)"):
                problem_type = "classification"
                y = pd.cut(y, bins=min(10, y.nunique()), labels=False)
                st.info("Target dikonversi ke kategori menggunakan binning")

        feature_columns = [col for col in df.columns if col != target_column]
        suitable_features = [col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() < 50]

        if len(suitable_features) < len(feature_columns):
            st.warning(f"⚠️ {len(feature_columns) - len(suitable_features)} kolom diabaikan karena terlalu banyak kategori")

        selected_features = st.multiselect("Pilih fitur (X):", suitable_features, default=suitable_features[:10])

        if selected_features:
            X = df[selected_features]
            X_processed, y_processed, label_encoder = preprocess_data(X, y, problem_type)

            max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
            min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1)
            max_depth_display = st.sidebar.slider("Max Depth Display", 1, 10, 3)
            split_ratio = st.slider("Test size (%)", 10, 50, 20)

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=split_ratio / 100, random_state=42
                )

                model = DecisionTreeClassifier(max_depth=max_depth,
                                               min_samples_split=min_samples_split,
                                               min_samples_leaf=min_samples_leaf,
                                               random_state=42) if problem_type == "classification" else                         DecisionTreeRegressor(max_depth=max_depth,
                                              min_samples_split=min_samples_split,
                                              min_samples_leaf=min_samples_leaf,
                                              random_state=42)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.subheader("📊 Evaluasi Model")
                if problem_type == "classification":
                    acc = accuracy_score(y_test, y_pred)
                    st.metric("Akurasi", f"{acc:.3f}")
                    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
                    st.dataframe(report_df.round(3))
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.metric("MSE", f"{mse:.3f}")
                    st.metric("R² Score", f"{r2:.3f}")

                st.subheader("🌳 Visualisasi Decision Tree")
                fig = plot_optimized_tree(model, X_processed.columns[:100], 
                                          label_encoder.classes_ if label_encoder else None,
                                          max_depth_display)
                st.pyplot(fig)
                plt.close()

                st.subheader("🔮 Prediksi Data Baru")
                input_data = {}
                with st.expander("Input Data untuk Prediksi"):
                    for col in selected_features:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                        else:
                            input_data[col] = st.selectbox(col, df[col].unique())

                if st.button("🎯 Prediksi"):
                    input_df = pd.DataFrame([input_data])
                    input_processed = pd.get_dummies(input_df, drop_first=True)
                    for col in X_processed.columns:
                        if col not in input_processed.columns:
                            input_processed[col] = 0
                    extra_cols = set(input_processed.columns) - set(X_processed.columns)
                    input_processed.drop(columns=extra_cols, inplace=True)
                    input_processed = input_processed[X_processed.columns]
                    pred = model.predict(input_processed)[0]
                    if label_encoder and problem_type == "classification":
                        pred = label_encoder.inverse_transform([pred])[0]
                    st.success(f"📌 Hasil prediksi: **{pred}**")

            except Exception as e:
                st.error(f"Error saat training/prediksi: {e}")

else:
    st.info("👆 Upload file untuk memulai")
