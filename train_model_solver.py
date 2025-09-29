import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset baru dengan target Resiko_Kelulusan
df = pd.read_csv("./dataset/ml_cleaned_resiko_dataset.csv")

# Fitur yang digunakan
feature_columns = [
    "IPK",
    "Mata Kuliah Tidak Lulus",
    "Jumlah Cuti Akademik",
    "Pekerjaan Sambil Kuliah",
    "Jumlah Semester",
    "IPS Rata-rata",
    "IPS Semester Akhir",
    "Performa IPS",
    "Kategori Kehadiran",
    "Lancar"
]

X = df[feature_columns]
y = df["Resiko_Kelulusan"]

# Bagi data train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Latih model
model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=4,
    criterion="entropy",
    random_state=42
)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print("✅ Akurasi:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Simpan model
joblib.dump(model, "./dataset/decision_tree_model_resiko.pkl")
print("✅ Model disimpan ke './dataset/decision_tree_model_resiko.pkl'")
