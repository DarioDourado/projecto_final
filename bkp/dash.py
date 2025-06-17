

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# =====================
# Carregamento de dados
# =====================

@st.cache_data
def load_data():
    paths = [
        'bkp/4-Carateristicas_salario.csv',
        'data/raw/4-Carateristicas_salario.csv',
        '4-Carateristicas_salario.csv'
    ]
    for path in paths:
        try:
            df = pd.read_csv(path)
            return df.dropna(), path
        except:
            continue
    return pd.DataFrame(), "Dados não encontrados"

# =====================
# DBSCAN Clustering
# =====================

def show_dbscan_clustering(df):
    st.title("🔬 Clustering com DBSCAN + PCA")
    df_num = df.select_dtypes(include=[np.number]).dropna()

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_num)

    st.sidebar.subheader("Parâmetros do DBSCAN")
    eps = st.sidebar.slider("eps", 0.1, 10.0, 1.5, 0.1)
    min_samples = st.sidebar.slider("min_samples", 2, 20, 5, 1)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_pca)

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    pct_noise = n_noise / len(clusters) if len(clusters) > 0 else 0
    st.metric("Clusters", n_clusters)
    st.metric("Ruído", n_noise)
    st.metric("% Ruído", f"{pct_noise:.1%}")

    score = silhouette_score(X_pca, clusters) if len(set(clusters)) > 1 else 0
    st.metric("📈 Silhouette Score", f"{score:.4f}")
    if len(set(clusters)) <= 1:
        st.warning("Silhouette Score não é válido para menos de 2 clusters.")
    st.caption("Silhouette Score mede a separação entre clusters (quanto mais perto de 1, melhor).")

    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=clusters.astype(str),
        title="Visualização dos Clusters DBSCAN",
        labels={"x": "PCA 1", "y": "PCA 2"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Download dos clusters
    output_csv = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
    output_csv["cluster"] = clusters
    csv_bytes = output_csv.to_csv(index=False).encode("utf-8")
    st.download_button("Download dos Clusters (csv)", data=csv_bytes, file_name="dbscan_clusters.csv", mime="text/csv")

# =====================
# Regras de Associação
# =====================

@st.cache_data
def load_csv(nome):
    try:
        df = pd.read_csv(f"output/analysis/{nome}.csv")
        return df
    except:
        return pd.DataFrame()

def show_association_rules():
    st.title("🧠 Regras de Associação")
    algo = st.selectbox("Algoritmo:", ["Apriori", "FP-Growth", "ECLAT"])
    nome_arquivo = {
        "Apriori": "apriori_rules",
        "FP-Growth": "fp_growth_rules",
        "ECLAT": "eclat_rules"
    }[algo]

    df = load_csv(nome_arquivo)
    if not df.empty:
        st.info(f"Número de regras encontradas: {len(df)}")
        # Filtros interativos
        min_lift = st.slider("Lift mínimo", float(df["lift"].min()), float(df["lift"].max()), float(df["lift"].min()))
        min_conf = st.slider("Confiança mínima", float(df["confidence"].min()), float(df["confidence"].max()), float(df["confidence"].min()))
        min_support = st.slider("Suporte mínimo", float(df["support"].min()), float(df["support"].max()), float(df["support"].min()))
        df = df[(df["lift"] >= min_lift) & (df["confidence"] >= min_conf) & (df["support"] >= min_support)]
        st.dataframe(df.sort_values("lift", ascending=False).head(10), use_container_width=True)
        # Download das regras filtradas
        csv_rules = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download das Regras Filtradas (csv)", data=csv_rules, file_name=f"{nome_arquivo}_filtrado.csv", mime="text/csv")
    else:
        st.warning("⚠️ Nenhuma regra encontrada.")

# =====================
# Main
# =====================

def main():
    st.set_page_config("Dashboard Científico", layout="wide")
    df, status = load_data()

    # Métricas resumo para sidebar
    silhouette = None
    n_regras = 0
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import DBSCAN
        df_num = df.select_dtypes(include=[np.number]).dropna()
        if not df_num.empty:
            X_pca = PCA(n_components=2).fit_transform(df_num)
            clusters = DBSCAN(eps=1.5, min_samples=5).fit_predict(X_pca)
            if len(set(clusters)) > 1:
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(X_pca, clusters)
    except:
        silhouette = None

    for rule_file in ["apriori_rules", "fp_growth_rules", "eclat_rules"]:
        regras = load_csv(rule_file)
        n_regras += len(regras)

    st.sidebar.title("📚 Menu")
    if silhouette is not None:
        st.sidebar.info(f"Silhouette Score (último clustering): {silhouette:.3f}")
    st.sidebar.info(f"Número total de regras de associação: {n_regras}")
    pages = [
        "📊 Visão Geral",
        "🔬 Clustering DBSCAN",
        "🧠 Regras de Associação"
    ]
    page = st.sidebar.radio("Ir para:", pages)

    if page == "📊 Visão Geral":
        st.title("📊 Visão Geral")
        st.write(f"Fonte: {status}")
        st.dataframe(df.head(50))
    elif page == "🔬 Clustering DBSCAN":
        show_dbscan_clustering(df)
    elif page == "🧠 Regras de Associação":
        show_association_rules()

if __name__ == "__main__":
    main()
