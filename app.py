import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Dashboard Ikan", layout="wide")

# --------------------------
# Login page
# --------------------------
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state["logged_in"] = True
        else:
            st.error("Login gagal")

# --------------------------
# Visualisasi: Pengaruh Ikan
# --------------------------
def pengaruh_ikan_dashboard():
    st.title("Dashboard K-Means Clustering")
    

    df = pd.read_excel("data bersih1.xlsx")

    # Filter Data di dalam halaman utama, bukan sidebar
    # Filter Data di sidebar
    st.sidebar.markdown("### Filter Data Dashboard")
    tahun_list = sorted(df["Tahun"].unique())
    tahun_filter = st.sidebar.selectbox("Pilih Tahun", tahun_list, key="tahun_filter_sidebar")

    kota_list = df[df["Tahun"] == tahun_filter]["Kab / Kota"].unique()
    kota_filter = st.sidebar.selectbox("Pilih Kabupaten/Kota", kota_list, key="kota_filter_sidebar")


    df_filtered = df[(df["Tahun"] == tahun_filter) & (df["Kab / Kota"] == kota_filter)]

    # Metrics
    st.markdown("### Ringkasan Data")
    total_produksi = df_filtered["Jumlah produksi ikan (ton)"].sum()
    total_nelayan = df_filtered["Jumlah nelayan laut"].sum()
    total_kapal = df_filtered["jumlah kapal"].sum()
    total_rt_perikanan = df_filtered["Rumah Tangga Perikanan"].sum()

    harga_cols = [
        "Harga Rata-Rata Tertimbang (Rp/kg)",
        "Harga Rata-Rata Tertimbang (Rp/kg).1",
        "Harga Rata-Rata Tertimbang (Rp/kg).2"
    ]
    harga_values = pd.to_numeric(df_filtered[harga_cols].iloc[0], errors="coerce")
    harga_values = harga_values[harga_values > 0]
    avg_harga = harga_values.mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Produksi Ikan (ton)", f"{total_produksi:,.2f}")
    with col2:
        st.metric("Jumlah Nelayan", f"{total_nelayan:,}")
    with col3:
        st.metric("Jumlah Kapal", f"{total_kapal:,}")
    with col4:
        st.metric("Rumah Tangga Perikanan", f"{total_rt_perikanan:,}")

    # Bar + Pie Chart
    col_bar, col_pie = st.columns([2, 1])

    with col_bar:
        st.subheader("Perbandingan Faktor Produksi")
        faktor_df = pd.DataFrame({
            "Faktor": ["Jumlah Nelayan", "Rumah Tangga Perikanan", "Jumlah Kapal", "Harga Rata-Rata (Rp/kg)"],
            "Nilai": [total_nelayan, total_rt_perikanan, total_kapal, avg_harga]
        })
        fig_bar = px.bar(faktor_df, x="Nilai", y="Faktor", orientation="h",
                         color="Faktor", title="Perbandingan Faktor Produksi")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_pie:
        st.subheader("Proporsi Produksi per Jenis Ikan")
        tongkol = df_filtered["Volume (ton)"].sum()
        tenggiri = df_filtered["Volume (ton).1"].sum()
        teri = df_filtered["Volume (ton).2"].sum()
        pie_df = pd.DataFrame({
            "Jenis Ikan": ["Tongkol", "Tenggiri", "Teri"],
            "Volume": [tongkol, tenggiri, teri]
        })
        fig_pie = px.pie(pie_df, names="Jenis Ikan", values="Volume", title="Distribusi Produksi")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Scatter plot clustering
        # Visualisasi PCA Interaktif
    st.markdown("### Visualisasi PCA Berdasarkan Hasil Clustering")

    # Data klaster
    data_klaster = pd.DataFrame({
        "Kab / Kota": ["Cianjur", "Garut", "Indramayu", "Karawang", "Kota Cirebon", "Pangandaran", "Subang", "Sukabumi", "Tasikmalaya", "Bekasi"],
        "Jumlah produksi ikan (ton)": [16.29, 1104.10, 83221.03, 1218.62, 162.30, 181.38, 1194.94, 141.57, 72.95, 139.80],
        "jumlah kapal": [962, 1062, 4583, 1332, 655, 1593, 965, 6, 253, 1217],
        "Jumlah nelayan laut": [1119, 4833, 41656, 500, 1564, 544, 3243, 204, 1670, 4022],
        "Harga Rata-Rata Gabungan (Rp/kg)": [22157.75, 20993.74, 30641.99, 15596.62, 44740.05, 38386.07, 44641.25, 23863.41, 30812.34, 41686.00],
        "Rumah Tangga Perikanan": [2850, 1089, 33347, 5640, 4164, 1672, 2293, 9103, 943, 1771]
    })

    features = [
        "Jumlah produksi ikan (ton)",
        "jumlah kapal",
        "Jumlah nelayan laut",
        "Harga Rata-Rata Gabungan (Rp/kg)",
        "Rumah Tangga Perikanan"
    ]

    # Standardisasi dan PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_klaster[features])

    kmeans = KMeans(n_clusters=3, random_state=42)
    data_klaster["Cluster"] = kmeans.fit_predict(scaled_data)

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_data)
    data_klaster["PCA1"] = components[:, 0]
    data_klaster["PCA2"] = components[:, 1]

    # Plot interaktif
    fig_pca = px.scatter(
        data_klaster,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        hover_name="Kab / Kota",
        title="PCA Visualisasi Clustering",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_pca, use_container_width=True)


# --------------------------
# Halaman KMeans dan PCA
# --------------------------
def load_data():
    return pd.DataFrame({
        'Kab / Kota': ['Cianjur', 'Garut', 'Indramayu', 'Karawang', 'Kota Cirebon', 'Pangandaran', 'Subang', 'Sukabumi', 'Tasikmalaya', 'Bekasi'],
        'Jumlah produksi ikan (ton)': [16.29, 1104.10, 83221.03, 1218.62, 162.30, 181.38, 1194.94, 141.57, 72.95, 139.80],
        'jumlah kapal': [962, 1062, 4583, 1332, 655, 1593, 965, 6, 253, 1217],
        'Jumlah nelayan laut': [1119, 4833, 41656, 500, 1564, 544, 3243, 204, 1670, 4022],
        'Harga Rata-Rata Gabungan (Rp/kg)': [22157.75, 20993.74, 30641.99, 15596.62, 44740.05, 38386.07, 44641.25, 23863.41, 30812.34, 41686.00],
        'Rumah Tangga Perikanan': [2850, 1089, 33347, 5640, 4164, 1672, 2293, 9103, 943, 1771]
    })

def kmeans_page():
    st.title("Pemodelan K-Means & Rekomendasi")
    df = load_data()

    st.subheader("Data Per Wilayah")
    st.dataframe(df)

    features = ['Jumlah produksi ikan (ton)', 'jumlah kapal', 'Jumlah nelayan laut', 'Harga Rata-Rata Gabungan (Rp/kg)', 'Rumah Tangga Perikanan']
    df_cluster = df.copy()

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster[features])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_cluster['Cluster'] = kmeans.fit_predict(df_scaled)

    cluster_summary = df_cluster.groupby('Cluster')[features].agg(['mean', 'min', 'max']).round(2)
    st.subheader("Statistik Rata-rata per Cluster")
    st.dataframe(cluster_summary)

    st.subheader("Wilayah & Cluster")
    st.dataframe(df_cluster[['Kab / Kota'] + features + ['Cluster']])

    st.subheader("🔍 Filter Rekomendasi Per Wilayah")
    wilayah = st.selectbox("Pilih Wilayah:", df_cluster['Kab / Kota'].unique())
    wilayah_data = df_cluster[df_cluster['Kab / Kota'] == wilayah].reset_index(drop=True)

    if not wilayah_data.empty:
        st.write(f"**Wilayah**: {wilayah}")
        st.write(f"**Masuk Cluster**: {int(wilayah_data.at[0, 'Cluster'])}")
        st.dataframe(wilayah_data[features])

        cluster_id = int(wilayah_data.at[0, 'Cluster'])
        st.write("**Rekomendasi:**")
        if cluster_id == 0:
            st.markdown("- Rekomendasi: **peningkatan jumlah kapal & akses pasar**.")
        elif cluster_id == 1:
            st.markdown("- Rekomendasi: **pelatihan & perluasan SDM**.")
        elif cluster_id == 2:
            st.markdown("- Rekomendasi: **fokus pada efisiensi dan ekspor**.")

def pca_visual_page():
    st.title("Visualisasi PCA dan Distribusi Cluster")
    df = load_data()
    features = ['Jumlah produksi ikan (ton)', 'jumlah kapal', 'Jumlah nelayan laut', 'Harga Rata-Rata Gabungan (Rp/kg)', 'Rumah Tangga Perikanan']

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)

    pca = PCA(n_components=2)
    components_ = pca.fit_transform(df_scaled)
    df['PCA1'] = components_[:, 0]
    df['PCA2'] = components_[:, 1]

    st.subheader("Plot PCA 2D")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100)
    for i, row in df.iterrows():
        ax.text(row['PCA1']+0.1, row['PCA2'], row['Kab / Kota'], fontsize=9)
    st.pyplot(fig)

    st.subheader("Distribusi Cluster per Variabel")
    for col in features:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Cluster', y=col, palette='Set2', ax=ax)
        st.pyplot(fig)

# --------------------------
# App Layout
# --------------------------
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login()
    else:
        st.sidebar.title("Navigasi")
        menu = st.sidebar.radio("Pilih Menu", (
            "Dashboard: Pengaruh Ikan",
            "Pemodelan K-Means"
           
        ))

        if menu == "Dashboard: Pengaruh Ikan":
            pengaruh_ikan_dashboard()
        elif menu == "Pemodelan K-Means":
            kmeans_page()

if __name__ == '__main__':
    main()
