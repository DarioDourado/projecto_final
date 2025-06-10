"""
🎯 Página de Clustering
Análise de agrupamentos com diferentes algoritmos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def show_clustering_page(df, i18n, charts):
    """Mostrar página de clustering"""
    
    if df is None or len(df) == 0:
        st.warning(i18n.t("messages.pipeline_needed"))
        return
    
    st.markdown(f"## {i18n.t('navigation.clustering')}")
    
    # Verificar se temos colunas numéricas suficientes
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Necessárias pelo menos 2 colunas numéricas para clustering")
        return
    
    # Tabs para diferentes análises
    tab1, tab2, tab3, tab4 = st.tabs([
        f"🎯 {i18n.t('clustering.execute')}",
        f"📊 {i18n.t('clustering.quality_metrics')}",
        f"📈 {i18n.t('clustering.cluster_statistics')}",
        "🔍 Análise Avançada"
    ])
    
    with tab1:
        show_clustering_interface(df, numeric_cols, i18n)
    
    with tab2:
        show_clustering_metrics(df, numeric_cols, i18n)
    
    with tab3:
        show_cluster_statistics(df, i18n)
    
    with tab4:
        show_advanced_clustering_analysis(df, numeric_cols, i18n)


def show_clustering_interface(df, numeric_cols, i18n):
    """Interface principal de clustering"""
    st.markdown(f"### 🎯 {i18n.t('clustering.execute')}")
    
    with st.form("clustering_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            algorithm = st.selectbox(
                f"🤖 {i18n.t('clustering.algorithm')}:",
                ["K-Means", "DBSCAN", "Hierarchical"]
            )
        
        with col2:
            if algorithm == "K-Means":
                n_clusters = st.slider(
                    f"📊 {i18n.t('clustering.num_clusters')}:",
                    2, 10, 3
                )
            elif algorithm == "DBSCAN":
                eps = st.slider("🎯 Eps (raio):", 0.1, 2.0, 0.5, 0.1)
                min_samples = st.slider("📊 Min Samples:", 2, 20, 5)
            else:  # Hierarchical
                n_clusters = st.slider(
                    f"📊 {i18n.t('clustering.num_clusters')}:",
                    2, 10, 3
                )
        
        with col3:
            selected_features = st.multiselect(
                f"🎯 {i18n.t('clustering.features')}:",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
        
        # Opções avançadas
        with st.expander("⚙️ Opções Avançadas"):
            normalize_data = st.checkbox("📊 Normalizar dados", value=True)
            use_pca = st.checkbox("🔍 Aplicar PCA", value=False)
            
            if use_pca:
                pca_components = st.slider("📊 Componentes PCA:", 2, min(5, len(selected_features)), 2)
        
        execute_button = st.form_submit_button(f"🚀 {i18n.t('clustering.execute')}")
        
        if execute_button:
            if not selected_features:
                st.error("Selecione pelo menos uma feature!")
                return
            
            execute_clustering(
                df, selected_features, algorithm, 
                locals(), i18n, normalize_data, use_pca
            )


def execute_clustering(df, features, algorithm, params, i18n, normalize_data, use_pca):
    """Executar algoritmo de clustering"""
    
    with st.spinner(f"{i18n.t('data.processing')}..."):
        try:
            # Preparar dados
            X = df[features].dropna()
            
            if len(X) == 0:
                st.error("Nenhum dado válido após remoção de valores faltantes")
                return
            
            # Normalização
            if normalize_data:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_for_clustering = X_scaled
            else:
                X_for_clustering = X.values
            
            # PCA se solicitado
            if use_pca and 'pca_components' in params:
                pca = PCA(n_components=params['pca_components'])
                X_for_clustering = pca.fit_transform(X_for_clustering)
                
                # Mostrar variância explicada
                explained_variance = pca.explained_variance_ratio_
                st.info(f"📊 Variância explicada pelo PCA: {explained_variance.sum():.2%}")
            
            # Executar clustering
            if algorithm == "K-Means":
                clusterer = KMeans(n_clusters=params['n_clusters'], random_state=42, n_init=10)
            elif algorithm == "DBSCAN":
                clusterer = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
            else:  # Hierarchical
                clusterer = AgglomerativeClustering(n_clusters=params['n_clusters'])
            
            clusters = clusterer.fit_predict(X_for_clustering)
            
            # Calcular métricas
            unique_clusters = len(np.unique(clusters))
            
            if unique_clusters > 1:
                silhouette_avg = silhouette_score(X_for_clustering, clusters)
                calinski_score = calinski_harabasz_score(X_for_clustering, clusters)
                
                # Mostrar resultados
                st.success(f"✅ {i18n.t('clustering.execute')} concluído!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(f"📊 {i18n.t('clustering.num_clusters')}", unique_clusters)
                
                with col2:
                    st.metric(f"🎯 {i18n.t('clustering.silhouette_score')}", f"{silhouette_avg:.3f}")
                
                with col3:
                    st.metric(f"📈 {i18n.t('clustering.points_processed')}", len(X))
                
                # Adicionar clusters ao DataFrame para visualização
                X_plot = X.copy()
                X_plot['Cluster'] = clusters
                
                # Visualizações
                create_clustering_visualizations(X_plot, features, i18n, use_pca, X_for_clustering, clusters)
                
                # Armazenar resultados na sessão
                st.session_state.clustering_results = {
                    'data': X_plot,
                    'algorithm': algorithm,
                    'features': features,
                    'silhouette_score': silhouette_avg,
                    'calinski_score': calinski_score,
                    'n_clusters': unique_clusters
                }
                
            else:
                st.warning("⚠️ Clustering resultou em apenas um grupo. Ajuste os parâmetros.")
                
        except Exception as e:
            st.error(f"Erro durante clustering: {e}")


def create_clustering_visualizations(data, features, i18n, use_pca, X_transformed, clusters):
    """Criar visualizações dos resultados de clustering"""
    
    st.markdown("### 📊 Visualizações dos Clusters")
    
    if use_pca and X_transformed.shape[1] >= 2:
        # Scatter plot PCA
        pca_df = pd.DataFrame(
            X_transformed[:, :2], 
            columns=['PC1', 'PC2']
        )
        pca_df['Cluster'] = clusters
        
        fig_pca = px.scatter(
            pca_df, x='PC1', y='PC2', 
            color='Cluster',
            title="🔍 Clusters no Espaço PCA",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig_pca.update_layout(
            title=dict(x=0.5, font=dict(size=16)),
            plot_bgcolor='rgba(248,249,250,0.8)'
        )
        
        st.plotly_chart(fig_pca, use_container_width=True)
    
    # Scatter plot com features originais
    if len(features) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("Feature X:", features, key="cluster_x")
        
        with col2:
            y_feature = st.selectbox("Feature Y:", features, index=1, key="cluster_y")
        
        if x_feature != y_feature:
            fig_original = px.scatter(
                data, x=x_feature, y=y_feature,
                color='Cluster',
                title=f"📊 Clusters: {x_feature} vs {y_feature}",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig_original.update_layout(
                title=dict(x=0.5, font=dict(size=16)),
                plot_bgcolor='rgba(248,249,250,0.8)'
            )
            
            st.plotly_chart(fig_original, use_container_width=True)
    
    # Distribuição dos clusters
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    
    fig_dist = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        title="📊 Distribuição dos Clusters",
        labels={'x': 'Cluster', 'y': 'Número de Pontos'},
        color=cluster_counts.values,
        color_continuous_scale='viridis'
    )
    
    fig_dist.update_layout(
        title=dict(x=0.5, font=dict(size=16)),
        plot_bgcolor='rgba(248,249,250,0.8)',
        showlegend=False
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)


def show_clustering_metrics(df, numeric_cols, i18n):
    """Mostrar métricas de qualidade do clustering"""
    st.markdown(f"### 📊 {i18n.t('clustering.quality_metrics')}")
    
    if len(numeric_cols) < 2:
        st.warning("Necessárias pelo menos 2 features numéricas")
        return
    
    st.markdown("#### 🎯 Análise do Número Ideal de Clusters (Método Elbow)")
    
    # Seleção de features para análise
    selected_features = st.multiselect(
        "Selecionar features para análise:",
        numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    )
    
    if not selected_features:
        st.warning("Selecione pelo menos uma feature")
        return
    
    if st.button("🚀 Executar Análise Elbow"):
        with st.spinner("Analisando diferentes números de clusters..."):
            try:
                # Preparar dados
                X = df[selected_features].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Método Elbow
                k_range = range(2, 11)
                inertias = []
                silhouette_scores = []
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(X_scaled)
                    inertias.append(kmeans.inertia_)
                    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
                
                # Gráfico Elbow
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_elbow = px.line(
                        x=list(k_range), y=inertias,
                        title="📊 Método Elbow - Inertia",
                        labels={'x': 'Número de Clusters', 'y': 'Inertia'},
                        markers=True
                    )
                    
                    fig_elbow.update_layout(
                        title=dict(x=0.5, font=dict(size=14)),
                        plot_bgcolor='rgba(248,249,250,0.8)'
                    )
                    
                    st.plotly_chart(fig_elbow, use_container_width=True)
                
                with col2:
                    fig_silhouette = px.line(
                        x=list(k_range), y=silhouette_scores,
                        title=f"🎯 {i18n.t('clustering.silhouette_score')}",
                        labels={'x': 'Número de Clusters', 'y': 'Silhouette Score'},
                        markers=True
                    )
                    
                    fig_silhouette.update_layout(
                        title=dict(x=0.5, font=dict(size=14)),
                        plot_bgcolor='rgba(248,249,250,0.8)'
                    )
                    
                    st.plotly_chart(fig_silhouette, use_container_width=True)
                
                # Recomendação
                best_k = k_range[np.argmax(silhouette_scores)]
                st.success(f"🎯 Número recomendado de clusters: {best_k} (maior Silhouette Score: {max(silhouette_scores):.3f})")
                
            except Exception as e:
                st.error(f"Erro na análise: {e}")


def show_cluster_statistics(df, i18n):
    """Mostrar estatísticas dos clusters"""
    st.markdown(f"### 📈 {i18n.t('clustering.cluster_statistics')}")
    
    # Verificar se temos resultados de clustering
    if 'clustering_results' not in st.session_state:
        st.info("Execute um clustering primeiro para ver as estatísticas")
        return
    
    results = st.session_state.clustering_results
    data = results['data']
    features = results['features']
    
    # Estatísticas por cluster
    st.markdown("#### 📊 Estatísticas por Cluster")
    
    cluster_stats = data.groupby('Cluster')[features].agg(['mean', 'std', 'count'])
    st.dataframe(cluster_stats, use_container_width=True)
    
    # Comparação visual entre clusters
    st.markdown("#### 📈 Comparação Visual entre Clusters")
    
    selected_feature = st.selectbox("Selecionar feature para comparação:", features)
    
    if selected_feature:
        # Box plot
        fig_box = px.box(
            data, x='Cluster', y=selected_feature,
            title=f"📊 Distribuição de {selected_feature} por Cluster"
        )
        
        fig_box.update_layout(
            title=dict(x=0.5, font=dict(size=16)),
            plot_bgcolor='rgba(248,249,250,0.8)'
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Histograma sobreposto
        fig_hist = px.histogram(
            data, x=selected_feature, color='Cluster',
            title=f"📊 Histograma de {selected_feature} por Cluster",
            barmode='overlay',
            opacity=0.7
        )
        
        fig_hist.update_layout(
            title=dict(x=0.5, font=dict(size=16)),
            plot_bgcolor='rgba(248,249,250,0.8)'
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)


def show_advanced_clustering_analysis(df, numeric_cols, i18n):
    """Análise avançada de clustering"""
    st.markdown("### 🔍 Análise Avançada")
    
    st.markdown("#### 🎯 Comparação de Algoritmos")
    
    if len(numeric_cols) < 2:
        st.warning("Necessárias pelo menos 2 features numéricas")
        return
    
    # Seleção de features
    selected_features = st.multiselect(
        "Selecionar features:",
        numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    )
    
    if not selected_features:
        st.warning("Selecione pelo menos uma feature")
        return
    
    if st.button("🚀 Comparar Algoritmos"):
        with st.spinner("Comparando algoritmos..."):
            try:
                # Preparar dados
                X = df[selected_features].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Definir algoritmos
                algorithms = {
                    'K-Means': KMeans(n_clusters=3, random_state=42, n_init=10),
                    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
                    'Hierarchical': AgglomerativeClustering(n_clusters=3)
                }
                
                results = []
                
                for name, algorithm in algorithms.items():
                    try:
                        labels = algorithm.fit_predict(X_scaled)
                        
                        if len(np.unique(labels)) > 1:
                            silhouette = silhouette_score(X_scaled, labels)
                            calinski = calinski_harabasz_score(X_scaled, labels)
                            n_clusters = len(np.unique(labels))
                            
                            results.append({
                                'Algoritmo': name,
                                'N° Clusters': n_clusters,
                                'Silhouette Score': silhouette,
                                'Calinski-Harabasz Score': calinski
                            })
                        else:
                            results.append({
                                'Algoritmo': name,
                                'N° Clusters': 1,
                                'Silhouette Score': 0,
                                'Calinski-Harabasz Score': 0
                            })
                    except Exception as e:
                        st.warning(f"Erro com algoritmo {name}: {e}")
                
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # Tabela de comparação
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Gráfico de comparação
                    fig_comparison = px.bar(
                        results_df, x='Algoritmo', y='Silhouette Score',
                        title="📊 Comparação de Algoritmos - Silhouette Score",
                        color='Silhouette Score',
                        color_continuous_scale='viridis'
                    )
                    
                    fig_comparison.update_layout(
                        title=dict(x=0.5, font=dict(size=16)),
                        plot_bgcolor='rgba(248,249,250,0.8)',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Recomendação
                    best_algorithm = results_df.loc[results_df['Silhouette Score'].idxmax(), 'Algoritmo']
                    best_score = results_df['Silhouette Score'].max()
                    
                    st.success(f"🏆 Melhor algoritmo: {best_algorithm} (Silhouette Score: {best_score:.3f})")
                
            except Exception as e:
                st.error(f"Erro na comparação: {e}")