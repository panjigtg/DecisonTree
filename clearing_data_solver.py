import pandas as pd

# Load dataset asli
df = pd.read_csv("./dataset/dataset_kelulusan_mahasiswa.csv")
df_clean = df.copy()

# Encode kategori ke numerik
df_clean["Pekerjaan Sambil Kuliah"] = df_clean["Pekerjaan Sambil Kuliah"].map({"Ya": 1, "Tidak": 0})
df_clean["Kategori Kehadiran"] = df_clean["Kategori Kehadiran"].map({"Rendah": 0, "Sedang": 1, "Tinggi": 2})

# Hapus NaN jika ada
df_clean.dropna(inplace=True)

# Tambahkan fitur baru
df_clean["Performa IPS"] = df_clean["IPS Semester Akhir"] - df_clean["IPS Rata-rata"]
df_clean["Lancar"] = (df_clean["Jumlah Semester"] <= 8).astype(int)

# Buat kolom target baru: Resiko_Kelulusan
def tentukan_resiko(row):
    skor = 0
    if row["IPK"] >= 3.0: skor += 1
    elif row["IPK"] < 2.5: skor -= 1

    if row["Mata Kuliah Tidak Lulus"] <= 2: skor += 1
    elif row["Mata Kuliah Tidak Lulus"] >= 5: skor -= 1

    if row["Jumlah Semester"] <= 8: skor += 1
    elif row["Jumlah Semester"] >= 13: skor -= 1

    if row["IPS Semester Akhir"] >= 3.0: skor += 1
    elif row["IPS Semester Akhir"] < 2.5: skor -= 1

    if row["IPS Tren"] > 0: skor += 1
    elif row["IPS Tren"] < -0.5: skor -= 1

    if row["Kategori Kehadiran"] == 2: skor += 1  # Tinggi
    elif row["Kategori Kehadiran"] == 0: skor -= 1  # Rendah

    if skor >= 3:
        return "Rendah"
    elif skor <= -1:
        return "Tinggi"
    else:
        return "Sedang"

df_clean["Resiko_Kelulusan"] = df_clean.apply(tentukan_resiko, axis=1)

# Simpan ke file baru
ml_cleaned_path = "./dataset/ml_cleaned_resiko_dataset.csv"
df_clean.to_csv(ml_cleaned_path, index=False)

print("✅ Dataset bersih disimpan ke:", ml_cleaned_path)
