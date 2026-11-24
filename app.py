import streamlit as st
import pandas as pd
import plotly.express as px
from src.model_engine import get_data, load_or_train

st.set_page_config(page_title="Akıllı Gayrimenkul Değerleme", page_icon="🏠", layout="wide")

st.title("🏠 Akıllı Gayrimenkul Değerleme & Pazar Analizi")
st.markdown("Konut özelliklerini girerek tahmini piyasa değerini, bölge karşılaştırmasını ve fiyatı belirleyen temel faktörleri inceleyin.")

@st.cache_resource
def init_system():
    df = get_data()
    model, metrics = load_or_train()
    return df, model, metrics

with st.spinner("Pazar verileri ve değerleme motoru hazırlanıyor..."):
    df, model, metrics = init_system()

# Sol Panel: Kullanıcı Girdileri
with st.sidebar:
    st.header("📋 Konut Özellikleri")
    gelir = st.slider("Bölge Gelir Seviyesi (Endeks)", 1.0, 15.0, 4.5, 0.5)
    yas = st.slider("Bina Yaşı", 1, 52, 20)
    oda = st.slider("Ortalama Oda Sayısı", 2.0, 10.0, 5.5, 0.5)
    yatak_odasi = st.slider("Yatak Odası Oranı", 0.5, 3.0, 1.0, 0.1)
    nufus = st.number_input("Bölge Nüfusu", 100, 10000, 1500, step=100)
    hane = st.slider("Hane Başına Kişi", 1.0, 6.0, 2.8, 0.2)
    
    st.markdown("---")
    st.subheader("📍 Konum Koordinatları")
    enlem = st.slider("Enlem (Latitude)", 32.5, 41.9, 34.05, 0.1)
    boylam = st.slider("Boylam (Longitude)", -124.3, -114.3, -118.25, 0.1)

# Tahmin Hesaplama
input_df = pd.DataFrame([{
    "Gelir_Endeksi": gelir,
    "Bina_Yasi": yas,
    "Ortalama_Oda": oda,
    "Yatak_Odasi": yatak_odasi,
    "Bolge_Nufusu": nufus,
    "Hane_Kisi_Sayisi": hane,
    "Enlem": enlem,
    "Boylam": boylam
}])

tahmin = model.predict(input_df)[0]
alt_sinir = max(0, tahmin - metrics["mae"])
ust_sinir = tahmin + metrics["mae"]

# Üst Metrik Kartları (Kesilmeyi önlemek için binlik 'K' formatı kullanıldı)
c1, c2, c3 = st.columns(3)
c1.metric("💰 Tahmini Piyasa Değeri", f"${tahmin:,.0f}")
c2.metric("📊 Güven Aralığı (±MAE)", f"${alt_sinir/1000:.0f}K - ${ust_sinir/1000:.0f}K")
c3.metric("🎯 Model Açıklayıcılığı (R²)", f"%{metrics['r2']*100:.1f}")

st.markdown("---")

# Alt Grafikler: Faktör Analizi ve Konum Haritası
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🔍 Fiyatı En Çok Ne Etkiliyor?")
    imp_df = pd.DataFrame({
        "Özellik": list(metrics["feature_importance"].keys()),
        "Etki_Payi": [v * 100 for v in metrics["feature_importance"].values()]
    }).sort_values("Etki_Payi", ascending=True)
    
    fig_imp = px.bar(imp_df, x="Etki_Payi", y="Özellik", orientation="h", 
                     labels={"Etki_Payi": "Fiyata Etki (%)", "Özellik": ""},
                     color="Etki_Payi", color_continuous_scale="Blues")
    fig_imp.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_imp, width="stretch")
with col_right:
    st.subheader("🗺️ Bölgesel Fiyat Dağılımı (Örneklem)")
    sample_df = df.sample(1000, random_state=42)
    fig_map = px.scatter_map(
        sample_df, lat="Enlem", lon="Boylam", color="Fiyat", size="Gelir_Endeksi",
        color_continuous_scale="Viridis", zoom=4.5, height=380,
        map_style="carto-darkmatter",
        labels={"Fiyat": "Fiyat ($)"}
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, width="stretch")