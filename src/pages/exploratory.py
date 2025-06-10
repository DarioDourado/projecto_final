"""
📈 Página de Análise Exploratória
Interface completa para exploração dos dados
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_exploratory_page(data, i18n):
    """Página principal de análise exploratória"""
    # Import safe
    try:
        from src.components.navigation import show_page_header
    except ImportError:
        def show_page_header(title, subtitle, icon):
            st.markdown(f"## {icon} {title}")
            if subtitle:
                st.markdown(f"*{subtitle}*")
    
    show_page_header(
        i18n.t('navigation.exploratory', 'Análise Exploratória'),
        i18n.t('exploratory.subtitle', 'Exploração detalhada dos dados e padrões'),
        "📈"
    )
    
    df = data.get('df')
    if df is None or len(df) == 0:
        st.warning(i18n.t('messages.pipeline_needed', 'Execute pipeline primeiro'))
        return
    
    # Tabs para diferentes análises
    tab1, tab2, tab3, tab4 = st.tabs([
        f"📊 {i18n.t('exploratory.distributions', 'Distribuições')}",
        f"🔗 {i18n.t('exploratory.correlations', 'Correlações')}",
        f"📈 {i18n.t('exploratory.interactive', 'Análise Interativa')}",
        f"📋 {i18n.t('exploratory.statistics', 'Estatísticas')}"
    ])
    
    with tab1:
        _show_distributions_analysis(df, i18n)
    
    with tab2:
        _show_correlation_analysis(df, i18n)
    
    with tab3:
        _show_interactive_analysis(df, i18n)
    
    with tab4:
        _show_statistical_analysis(df, i18n)

def _show_distributions_analysis(df, i18n):
    """Análise de distribuições"""
    st.subheader(f"📊 {i18n.t('exploratory.distributions_title', 'Análise de Distribuições')}")
    
    # Seletor de variável
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if numeric_cols:
            selected_numeric = st.selectbox(
                f"🔢 {i18n.t('exploratory.select_numeric', 'Selecionar Variável Numérica')}:",
                numeric_cols,
                key="numeric_dist"
            )
            
            if selected_numeric:
                _plot_numeric_distribution(df, selected_numeric, i18n)
    
    with col2:
        if categorical_cols:
            selected_categorical = st.selectbox(
                f"📝 {i18n.t('exploratory.select_categorical', 'Selecionar Variável Categórica')}:",
                categorical_cols,
                key="categorical_dist"
            )
            
            if selected_categorical:
                _plot_categorical_distribution(df, selected_categorical, i18n)

def _plot_numeric_distribution(df, column, i18n):
    """Plotar distribuição de variável numérica"""
    st.markdown(f"#### 📊 {column}")
    
    try:
        # Histograma com densidade
        fig = px.histogram(
            df, 
            x=column,
            nbins=30,
            title=f"Distribuição de {column}",
            marginal="box",
            color_discrete_sequence=['#667eea'],
            template="plotly_white"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas descritivas
        stats = df[column].describe()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Média", f"{stats['mean']:.2f}")
        with col2:
            st.metric("📈 Mediana", f"{stats['50%']:.2f}")
        with col3:
            st.metric("📏 Desvio", f"{stats['std']:.2f}")
        with col4:
            st.metric("📋 Missing", f"{df[column].isnull().sum()}")
            
    except Exception as e:
        st.error(f"Erro ao plotar {column}: {e}")

def _plot_categorical_distribution(df, column, i18n):
    """Plotar distribuição de variável categórica"""
    st.markdown(f"#### 📝 {column}")
    
    try:
        value_counts = df[column].value_counts()
        
        # Gráfico de barras
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Distribuição de {column}",
            color=value_counts.values,
            color_continuous_scale='viridis',
            template="plotly_white"
        )
        
        fig.update_layout(
            xaxis_tickangle=45,
            height=400,
            xaxis_title=column,
            yaxis_title="Contagem"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas categóricas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Únicos", f"{df[column].nunique()}")
        with col2:
            st.metric("🏆 Mais Frequente", f"{value_counts.index[0]}")
        with col3:
            st.metric("📈 Freq. Max", f"{value_counts.iloc[0]}")
        with col4:
            st.metric("📋 Missing", f"{df[column].isnull().sum()}")
            
    except Exception as e:
        st.error(f"Erro ao plotar {column}: {e}")

def _show_correlation_analysis(df, i18n):
    """Análise de correlações"""
    st.subheader(f"🔗 {i18n.t('exploratory.correlations_title', 'Análise de Correlações')}")
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        st.warning("Poucas variáveis numéricas para análise de correlação")
        return
    
    # Matriz de correlação
    try:
        corr_matrix = numeric_df.corr()
        
        # Heatmap interativo
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Matriz de Correlação",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlações
        _show_top_correlations(corr_matrix, i18n)
        
    except Exception as e:
        st.error(f"Erro na análise de correlação: {e}")

def _show_top_correlations(corr_matrix, i18n):
    """Mostrar top correlações"""
    st.markdown("#### 🏆 Top Correlações")
    
    try:
        # Remover correlações consigo mesmo e duplicatas
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                if not pd.isna(corr_value):
                    correlations.append({
                        'Variável 1': var1,
                        'Variável 2': var2,
                        'Correlação': corr_value,
                        'Abs Correlação': abs(corr_value)
                    })
        
        if correlations:
            corr_df = pd.DataFrame(correlations)
            corr_df = corr_df.sort_values('Abs Correlação', ascending=False).head(10)
            
            # Mostrar tabela
            st.dataframe(
                corr_df[['Variável 1', 'Variável 2', 'Correlação']].round(3),
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Erro ao calcular top correlações: {e}")

def _show_interactive_analysis(df, i18n):
    """Análise interativa"""
    st.subheader(f"📈 {i18n.t('exploratory.interactive_title', 'Análise Interativa')}")
    
    # Controles
    col1, col2, col3 = st.columns(3)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    with col1:
        if numeric_cols:
            x_var = st.selectbox("📊 Eixo X:", numeric_cols, key="x_var")
    
    with col2:
        y_options = ["Nenhuma"] + numeric_cols
        y_var = st.selectbox("📈 Eixo Y:", y_options, key="y_var")
    
    with col3:
        color_options = ["Nenhuma"] + categorical_cols
        color_var = st.selectbox("🎨 Cor por:", color_options, key="color_var")
    
    # Gerar gráfico baseado na seleção
    if 'x_var' in locals():
        try:
            if y_var == "Nenhuma":
                # Histograma
                fig = px.histogram(
                    df, 
                    x=x_var,
                    color=color_var if color_var != "Nenhuma" else None,
                    title=f"Distribuição de {x_var}",
                    template="plotly_white",
                    marginal="box"
                )
            else:
                # Scatter plot
                fig = px.scatter(
                    df, 
                    x=x_var, 
                    y=y_var,
                    color=color_var if color_var != "Nenhuma" else None,
                    title=f"{x_var} vs {y_var}",
                    template="plotly_white",
                    opacity=0.7
                )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro ao criar gráfico: {e}")

def _show_statistical_analysis(df, i18n):
    """Análise estatística"""
    st.subheader(f"📋 {i18n.t('exploratory.statistics_title', 'Estatísticas Descritivas')}")
    
    # Estatísticas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        st.markdown("#### 🔢 Variáveis Numéricas")
        st.dataframe(numeric_df.describe().round(3), use_container_width=True)
    
    # Estatísticas categóricas
    categorical_df = df.select_dtypes(include=['object'])
    if not categorical_df.empty:
        st.markdown("#### 📝 Variáveis Categóricas")
        
        cat_stats = []
        for col in categorical_df.columns:
            stats = {
                'Coluna': col,
                'Únicos': df[col].nunique(),
                'Mais Frequente': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                'Freq. Max': df[col].value_counts().iloc[0] if not df[col].empty else 0,
                'Missing': df[col].isnull().sum(),
                'Missing %': (df[col].isnull().sum() / len(df)) * 100
            }
            cat_stats.append(stats)
        
        if cat_stats:
            cat_df = pd.DataFrame(cat_stats)
            st.dataframe(cat_df, use_container_width=True)
    
    # Informações gerais do dataset
    st.markdown("#### 📊 Informações Gerais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Total Registros", f"{len(df):,}")
    
    with col2:
        st.metric("📊 Total Colunas", len(df.columns))
    
    with col3:
        total_missing = df.isnull().sum().sum()
        st.metric("❌ Total Missing", f"{total_missing:,}")
    
    with col4:
        missing_percent = (total_missing / (len(df) * len(df.columns))) * 100
        st.metric("📈 Missing %", f"{missing_percent:.1f}%")