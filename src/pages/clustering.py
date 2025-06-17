"""
Página de Clustering DBSCAN
Análise de Segmentação baseada em Densidade
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

def show_clustering_page(data):
    """Página de clustering DBSCAN"""
    st.title("🎯 Clustering DBSCAN")
    st.markdown("### Análise de Segmentação baseada em Densidade")
    
    # Introdução contextual
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
    ">
        <p style="margin: 0; color: #495057;">
            🎯 <strong>DBSCAN (Density-Based Spatial Clustering)</strong> é um algoritmo de clustering 
            que agrupa pontos densamente agrupados e marca pontos isolados como ruído (outliers).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'dbscan_results' not in data:
        show_no_clustering_data()
        return
    
    dbscan_df = data['dbscan_results']
    
    # Verificar se dados são válidos
    if dbscan_df.empty:
        st.warning("⚠️ Dataset de clustering vazio")
        return
    
    # Análise estatística dos clusters
    show_cluster_statistics(dbscan_df)
    
    # Visualizações dos clusters
    show_cluster_visualizations(dbscan_df)
    
    # Análise detalhada por cluster
    show_cluster_analysis(dbscan_df)
    
    # Métricas de qualidade
    show_clustering_metrics(dbscan_df)
    
    # Insights e recomendações
    show_clustering_insights(dbscan_df)

def show_no_clustering_data():
    """Mostrar aviso quando dados de clustering não estão disponíveis"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 2rem 0;
        border: 1px solid #ffecb5;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
        <h3 style="color: #856404; margin-bottom: 1rem;">Análise DBSCAN Não Executada</h3>
        <p style="color: #856404; margin-bottom: 1.5rem;">
            Execute o pipeline completo para gerar os resultados de clustering.
        </p>
        <div style="
            background: rgba(133, 100, 4, 0.1);
            padding: 1rem;
            border-radius: 8px;
            font-family: monospace;
            color: #856404;
            font-weight: bold;
        ">
            python main.py
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Informações sobre DBSCAN
    st.markdown("#### 📚 Sobre o Algoritmo DBSCAN")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Características:**
        - Baseado em densidade
        - Detecta clusters de forma arbitrária
        - Identifica outliers automaticamente
        - Não requer número de clusters predefinido
        """)
    
    with col2:
        st.markdown("""
        **⚙️ Parâmetros:**
        - **eps**: Distância máxima entre pontos
        - **min_samples**: Número mínimo de pontos
        - **metric**: Métrica de distância utilizada
        """)

def show_cluster_statistics(dbscan_df):
    """Mostrar estatísticas dos clusters"""
    st.subheader("📊 Estatísticas dos Clusters")
    
    if 'cluster' not in dbscan_df.columns:
        st.error("❌ Coluna 'cluster' não encontrada nos dados")
        return
    
    cluster_counts = dbscan_df['cluster'].value_counts().sort_index()
    n_clusters = len(cluster_counts[cluster_counts.index != -1])
    noise_points = cluster_counts.get(-1, 0)
    total_points = len(dbscan_df)
    
    # Cards de métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card(
            "🎯 Clusters Válidos",
            n_clusters,
            "Grupos identificados",
            "#28a745"
        )
    
    with col2:
        create_metric_card(
            "🔴 Pontos de Ruído",
            noise_points,
            "Outliers detectados",
            "#dc3545"
        )
    
    with col3:
        noise_rate = (noise_points / total_points) * 100 if total_points > 0 else 0
        create_metric_card(
            "📊 Taxa de Ruído",
            f"{noise_rate:.1f}%",
            "Percentual de outliers",
            "#ffc107"
        )
    
    with col4:
        avg_cluster_size = (total_points - noise_points) / n_clusters if n_clusters > 0 else 0
        create_metric_card(
            "👥 Tamanho Médio",
            f"{avg_cluster_size:.0f}",
            "Pontos por cluster",
            "#17a2b8"
        )

def create_metric_card(title, value, description, color):
    """Criar card de métrica estilizado"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}08);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid {color};
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <h4 style="margin: 0; color: #333; font-size: 0.9rem; font-weight: 600;">
            {title}
        </h4>
        <h2 style="margin: 0.8rem 0; color: {color}; font-size: 2rem; font-weight: bold;">
            {value}
        </h2>
        <p style="margin: 0; color: #666; font-size: 0.8rem; opacity: 0.8;">
            {description}
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_cluster_visualizations(dbscan_df):
    """Mostrar visualizações dos clusters"""
    st.subheader("📈 Visualizações dos Clusters")
    
    cluster_counts = dbscan_df['cluster'].value_counts().sort_index()
    
    # Duas visualizações lado a lado
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de barras da distribuição
        fig_bar = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            title="Distribuição dos Clusters",
            labels={'x': 'Cluster ID', 'y': 'Número de Pontos'},
            color=cluster_counts.index,
            color_continuous_scale='viridis'
        )
        
        # Destacar ruído em vermelho
        fig_bar.update_traces(
            marker_color=['red' if x == -1 else f'rgba(68, 133, 244, 0.8)' 
                         for x in cluster_counts.index]
        )
        
        fig_bar.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Cluster ID (-1 = Ruído)",
            yaxis_title="Quantidade de Pontos"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Gráfico de pizza (sem ruído para melhor visualização)
        valid_clusters = cluster_counts[cluster_counts.index != -1]
        
        if len(valid_clusters) > 0:
            fig_pie = px.pie(
                values=valid_clusters.values,
                names=[f'Cluster {i}' for i in valid_clusters.index],
                title="Proporção dos Clusters Válidos"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("⚠️ Nenhum cluster válido encontrado")
    
    # Visualização 2D se houver colunas numéricas suficientes
    numeric_cols = dbscan_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'cluster']
    
    if len(numeric_cols) >= 2:
        show_2d_cluster_plot(dbscan_df, numeric_cols)

def show_2d_cluster_plot(dbscan_df, numeric_cols):
    """Mostrar plot 2D dos clusters"""
    st.markdown("#### 📍 Visualização 2D dos Clusters")
    
    # Seletor de variáveis
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox(
            "Variável X:",
            numeric_cols,
            key="cluster_x_var"
        )
    
    with col2:
        y_vars = [col for col in numeric_cols if col != x_var]
        if y_vars:
            y_var = st.selectbox(
                "Variável Y:",
                y_vars,
                key="cluster_y_var"
            )
        else:
            y_var = None
    
    if x_var and y_var:
        # Criar scatter plot
        fig_scatter = px.scatter(
            dbscan_df,
            x=x_var,
            y=y_var,
            color='cluster',
            title=f"Clusters: {y_var} vs {x_var}",
            color_continuous_scale='viridis',
            hover_data=['cluster']
        )
        
        # Personalizar cores para ruído
        fig_scatter.update_traces(
            marker=dict(
                size=8,
                opacity=0.7,
                line=dict(width=1, color='white')
            )
        )
        
        fig_scatter.update_layout(
            height=500,
            xaxis_title=x_var.replace('_', ' ').title(),
            yaxis_title=y_var.replace('_', ' ').title()
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)

def show_cluster_analysis(dbscan_df):
    """Análise detalhada por cluster"""
    st.subheader("🔍 Análise Detalhada por Cluster")
    
    cluster_counts = dbscan_df['cluster'].value_counts().sort_index()
    
    # Criar tabela de detalhes
    cluster_details = []
    for cluster_id in sorted(cluster_counts.index):
        cluster_data = dbscan_df[dbscan_df['cluster'] == cluster_id]
        
        detail = {
            'Cluster ID': cluster_id,
            'Tipo': 'Ruído' if cluster_id == -1 else 'Válido',
            'Pontos': cluster_counts[cluster_id],
            'Percentual': f"{(cluster_counts[cluster_id] / len(dbscan_df)) * 100:.2f}%"
        }
        
        # Adicionar estatísticas das colunas numéricas
        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'cluster']
        
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:  # Limitar a 3 colunas para não sobrecarregar
                detail[f'{col}_mean'] = f"{cluster_data[col].mean():.2f}"
        
        cluster_details.append(detail)
    
    # Mostrar tabela
    details_df = pd.DataFrame(cluster_details)
    st.dataframe(details_df, use_container_width=True)
    
    # Análise individual dos clusters
    if len(cluster_counts) > 1:
        st.markdown("#### 📋 Perfil dos Clusters")
        
        selected_cluster = st.selectbox(
            "Selecionar cluster para análise detalhada:",
            sorted(cluster_counts.index),
            format_func=lambda x: f"Cluster {x}" if x != -1 else "Ruído (Outliers)"
        )
        
        show_individual_cluster_analysis(dbscan_df, selected_cluster)

def show_individual_cluster_analysis(dbscan_df, cluster_id):
    """Análise individual de um cluster específico"""
    cluster_data = dbscan_df[dbscan_df['cluster'] == cluster_id]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**📊 Informações do {'Ruído' if cluster_id == -1 else f'Cluster {cluster_id}'}:**")
        
        info = {
            "🔢 Total de Pontos": len(cluster_data),
            "📊 % do Dataset": f"{(len(cluster_data) / len(dbscan_df)) * 100:.2f}%",
            "🎯 Densidade": "Baixa" if cluster_id == -1 else "Alta"
        }
        
        for key, value in info.items():
            st.write(f"- {key}: {value}")
    
    with col2:
        # Estatísticas das colunas numéricas
        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'cluster']
        
        if len(numeric_cols) > 0:
            st.markdown("**📈 Estatísticas Numéricas:**")
            
            for col in numeric_cols[:3]:
                mean_val = cluster_data[col].mean()
                std_val = cluster_data[col].std()
                st.write(f"- {col}: μ={mean_val:.2f}, σ={std_val:.2f}")

def show_clustering_metrics(dbscan_df):
    """Mostrar métricas de qualidade do clustering"""
    st.subheader("📏 Métricas de Qualidade")
    
    try:
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        # Preparar dados para métricas (apenas clusters válidos)
        valid_data = dbscan_df[dbscan_df['cluster'] != -1]
        
        if len(valid_data) == 0:
            st.warning("⚠️ Nenhum cluster válido para calcular métricas")
            return
        
        numeric_cols = valid_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'cluster']
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Dados insuficientes para calcular métricas de qualidade")
            return
        
        X = valid_data[numeric_cols]
        labels = valid_data['cluster']
        
        # Calcular métricas
        if len(labels.unique()) > 1:
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
            
            # Mostrar métricas em cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Silhouette Score (-1 a 1, quanto maior melhor)
                silhouette_color = "#28a745" if silhouette > 0.5 else "#ffc107" if silhouette > 0.2 else "#dc3545"
                create_metric_card(
                    "📊 Silhouette Score",
                    f"{silhouette:.3f}",
                    "Qualidade da separação",
                    silhouette_color
                )
            
            with col2:
                # Calinski-Harabasz Score (quanto maior melhor)
                ch_color = "#28a745" if calinski > 100 else "#ffc107" if calinski > 50 else "#dc3545"
                create_metric_card(
                    "📈 Calinski-Harabasz",
                    f"{calinski:.1f}",
                    "Razão variância inter/intra",
                    ch_color
                )
            
            with col3:
                # Número efetivo de clusters
                n_clusters = len(labels.unique())
                create_metric_card(
                    "🎯 Clusters Efetivos",
                    n_clusters,
                    "Grupos bem definidos",
                    "#17a2b8"
                )
    
    except ImportError:
        st.info("📦 Instale scikit-learn para métricas avançadas: `pip install scikit-learn`")

def show_clustering_insights(dbscan_df):
    """Mostrar insights e recomendações"""
    st.subheader("💡 Insights e Recomendações")
    
    cluster_counts = dbscan_df['cluster'].value_counts().sort_index()
    n_clusters = len(cluster_counts[cluster_counts.index != -1])
    noise_points = cluster_counts.get(-1, 0)
    total_points = len(dbscan_df)
    noise_rate = (noise_points / total_points) * 100
    
    insights = []
    
    # Análise do número de clusters
    if n_clusters == 0:
        insights.append({
            "icon": "🔴",
            "title": "Nenhum Cluster Identificado",
            "message": "Todos os pontos foram classificados como ruído. Considere ajustar os parâmetros eps e min_samples.",
            "type": "error"
        })
    elif n_clusters == 1:
        insights.append({
            "icon": "🟡",
            "title": "Cluster Único Detectado",
            "message": "Apenas um cluster foi identificado. Os dados podem ser muito homogêneos ou os parâmetros muito restritivos.",
            "type": "warning"
        })
    elif 2 <= n_clusters <= 5:
        insights.append({
            "icon": "🟢",
            "title": "Segmentação Adequada",
            "message": f"{n_clusters} clusters identificados. Boa segmentação para análise e interpretação.",
            "type": "success"
        })
    else:
        insights.append({
            "icon": "🟡",
            "title": "Muitos Clusters Detectados",
            "message": f"{n_clusters} clusters podem indicar over-segmentação. Considere aumentar o parâmetro eps.",
            "type": "warning"
        })
    
    # Análise da taxa de ruído
    if noise_rate > 30:
        insights.append({
            "icon": "🔴",
            "title": "Alta Taxa de Ruído",
            "message": f"{noise_rate:.1f}% dos pontos são outliers. Parâmetros podem estar muito restritivos.",
            "type": "error"
        })
    elif noise_rate > 15:
        insights.append({
            "icon": "🟡",
            "title": "Taxa de Ruído Moderada",
            "message": f"{noise_rate:.1f}% de outliers detectados. Monitore a qualidade da segmentação.",
            "type": "warning"
        })
    else:
        insights.append({
            "icon": "🟢",
            "title": "Taxa de Ruído Controlada",
            "message": f"Apenas {noise_rate:.1f}% de outliers. Boa detecção de densidade.",
            "type": "success"
        })
    
    # Mostrar insights
    for insight in insights:
        bg_color = {
            "success": "#d4edda",
            "warning": "#fff3cd", 
            "error": "#f8d7da"
        }.get(insight["type"], "#e9ecef")
        
        border_color = {
            "success": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545"
        }.get(insight["type"], "#6c757d")
        
        st.markdown(f"""
        <div style="
            background: {bg_color};
            padding: 1.2rem;
            border-radius: 10px;
            border-left: 4px solid {border_color};
            margin: 1rem 0;
            display: flex;
            align-items: flex-start;
        ">
            <div style="font-size: 1.5rem; margin-right: 1rem; margin-top: 0.2rem;">
                {insight['icon']}
            </div>
            <div>
                <h4 style="margin: 0 0 0.5rem 0; color: #333;">
                    {insight['title']}
                </h4>
                <p style="margin: 0; color: #555; line-height: 1.5;">
                    {insight['message']}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recomendações técnicas
    st.markdown("#### 🔧 Recomendações Técnicas")
    
    recommendations = [
        "🎯 **Ajuste de Parâmetros**: Experimente diferentes valores de eps e min_samples",
        "📊 **Validação**: Use métricas como Silhouette Score para avaliar qualidade",
        "🔍 **Análise de Outliers**: Investigue os pontos classificados como ruído",
        "📈 **Interpretação**: Analise as características de cada cluster identificado"
    ]
    
    for rec in recommendations:
        st.markdown(f"- {rec}")