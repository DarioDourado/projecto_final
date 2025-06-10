"""
📈 Página de Análise Exploratória
Ferramentas interativas para exploração de dados
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from src.components.navigation import show_page_header, show_breadcrumbs
from src.components.layout import create_status_box

def show_exploratory_page(data, i18n):
    """Página de análise exploratória interativa"""
    
    # Header da página
    show_page_header(
        title=i18n.t('navigation.exploratory', 'Análise Exploratória'),
        subtitle=i18n.t('exploratory.subtitle', 'Explore os dados de forma interativa'),
        icon="📈"
    )
    
    # Breadcrumbs
    show_breadcrumbs([
        (i18n.t('navigation.overview', 'Visão Geral'), 'navigation.overview'),
        (i18n.t('navigation.exploratory', 'Análise Exploratória'), 'navigation.exploratory')
    ], i18n)
    
    df = data.get('df')
    if df is None or len(df) == 0:
        st.warning(i18n.t('messages.pipeline_needed', '⚠️ Execute: python main.py'))
        return
    
    # Tabs de análise
    tab1, tab2, tab3, tab4 = st.tabs([
        f"🔍 {i18n.t('exploratory.univariate', 'Análise Univariada')}",
        f"📊 {i18n.t('exploratory.bivariate', 'Análise Bivariada')}",
        f"🔗 {i18n.t('exploratory.correlations', 'Correlações')}",
        f"📋 {i18n.t('exploratory.data_quality', 'Qualidade dos Dados')}"
    ])
    
    with tab1:
        _show_univariate_analysis(df, i18n)
    
    with tab2:
        _show_bivariate_analysis(df, i18n)
    
    with tab3:
        _show_correlation_analysis(df, i18n)
    
    with tab4:
        _show_data_quality_analysis(df, i18n)

def _show_univariate_analysis(df, i18n):
    """Análise univariada - uma variável por vez"""
    st.markdown(f"### 🔍 {i18n.t('exploratory.select_variable', 'Selecione uma Variável')}")
    
    # Separar variáveis por tipo
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        variable_type = st.selectbox(
            "🎯 Tipo de Variável:",
            ["Numérica", "Categórica"]
        )
    
    with col2:
        if variable_type == "Numérica" and numeric_cols:
            selected_var = st.selectbox("📊 Variável Numérica:", numeric_cols)
        elif variable_type == "Categórica" and categorical_cols:
            selected_var = st.selectbox("📋 Variável Categórica:", categorical_cols)
        else:
            st.warning("Nenhuma variável deste tipo disponível")
            return
    
    if selected_var:
        col1, col2 = st.columns(2)
        
        with col1:
            # Estatísticas descritivas
            st.markdown("#### 📈 Estatísticas")
            if variable_type == "Numérica":
                stats = df[selected_var].describe()
                st.dataframe(stats, use_container_width=True)
            else:
                value_counts = df[selected_var].value_counts()
                st.dataframe(value_counts, use_container_width=True)
        
        with col2:
            # Gráfico
            st.markdown("#### 📊 Visualização")
            if variable_type == "Numérica":
                fig = px.histogram(
                    df, x=selected_var,
                    title=f"Distribuição: {selected_var}",
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
            else:
                value_counts = df[selected_var].value_counts()
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Contagem: {selected_var}",
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
            
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

def _show_bivariate_analysis(df, i18n):
    """Análise bivariada - relação entre duas variáveis"""
    st.markdown(f"### 📊 {i18n.t('exploratory.select_two_variables', 'Selecione Duas Variáveis')}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    all_cols = numeric_cols + categorical_cols
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        var1 = st.selectbox("🎯 Variável X:", all_cols, key="bivar_x")
    
    with col2:
        var2 = st.selectbox("🎯 Variável Y:", all_cols, key="bivar_y")
    
    with col3:
        color_var = st.selectbox(
            "🎨 Colorir por:", 
            ["Nenhum"] + categorical_cols,
            key="bivar_color"
        )
    
    if var1 and var2 and var1 != var2:
        # Determinar tipo de gráfico baseado nos tipos de variáveis
        var1_numeric = var1 in numeric_cols
        var2_numeric = var2 in numeric_cols
        
        color_column = None if color_var == "Nenhum" else color_var
        
        if var1_numeric and var2_numeric:
            # Scatter plot
            fig = px.scatter(
                df, x=var1, y=var2, color=color_column,
                title=f"Relação: {var1} vs {var2}",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
        elif var1_numeric and not var2_numeric:
            # Box plot
            fig = px.box(
                df, x=var2, y=var1, color=color_column,
                title=f"Distribuição de {var1} por {var2}",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
        elif not var1_numeric and var2_numeric:
            # Box plot invertido
            fig = px.box(
                df, x=var1, y=var2, color=color_column,
                title=f"Distribuição de {var2} por {var1}",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
        else:
            # Heatmap de crosstab
            crosstab = pd.crosstab(df[var1], df[var2], normalize='index') * 100
            fig = px.imshow(
                crosstab,
                title=f"Crosstab: {var1} vs {var2} (%)",
                template="plotly_white",
                color_continuous_scale="Blues",
                aspect="auto"
            )
        
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas da relação
        if var1_numeric and var2_numeric:
            correlation = df[var1].corr(df[var2])
            st.markdown(create_status_box(
                f"📊 Correlação entre {var1} e {var2}: {correlation:.3f}",
                "info"
            ), unsafe_allow_html=True)

def _show_correlation_analysis(df, i18n):
    """Análise de correlações"""
    st.markdown(f"### 🔗 {i18n.t('charts.correlation_matrix', 'Matriz de Correlação')}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Precisamos de pelo menos 2 variáveis numéricas para análise de correlação")
        return
    
    # Opções de configuração
    col1, col2, col3 = st.columns(3)
    
    with col1:
        method = st.selectbox(
            "🔧 Método de Correlação:",
            ["pearson", "spearman", "kendall"]
        )
    
    with col2:
        min_correlation = st.slider(
            "📊 Correlação Mínima:",
            0.0, 1.0, 0.0, 0.05
        )
    
    with col3:
        show_values = st.checkbox("🔢 Mostrar Valores", value=True)
    
    # Calcular matriz de correlação
    corr_matrix = df[numeric_cols].corr(method=method)
    
    # Filtrar correlações baixas se especificado
    if min_correlation > 0:
        mask = (np.abs(corr_matrix) >= min_correlation) | (corr_matrix == 1.0)
        corr_matrix = corr_matrix.where(mask)
    
    # Criar heatmap
    fig = px.imshow(
        corr_matrix,
        title=f"Matriz de Correlação ({method.title()})",
        template="plotly_white",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        aspect="auto"
    )
    
    if show_values:
        # Adicionar valores de texto
        fig.update_traces(
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10}
        )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        width=700,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top correlações
    st.markdown("#### 🏆 Correlações Mais Altas")
    
    # Extrair correlações em formato de lista
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            if not pd.isna(corr_value):
                corr_pairs.append({
                    'Variável 1': var1,
                    'Variável 2': var2,
                    'Correlação': corr_value,
                    'Correlação Absoluta': abs(corr_value)
                })
    
    if corr_pairs:
        corr_df = pd.DataFrame(corr_pairs)
        corr_df = corr_df.sort_values('Correlação Absoluta', ascending=False)
        
        # Mostrar top 10
        st.dataframe(
            corr_df.head(10)[['Variável 1', 'Variável 2', 'Correlação']], 
            use_container_width=True
        )

def _show_data_quality_analysis(df, i18n):
    """Análise da qualidade dos dados"""
    st.markdown(f"### 📋 {i18n.t('exploratory.data_quality', 'Qualidade dos Dados')}")
    
    # Resumo geral
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total de Registros", f"{len(df):,}")
    
    with col2:
        st.metric("📋 Total de Colunas", len(df.columns))
    
    with col3:
        total_missing = df.isnull().sum().sum()
        st.metric("❌ Valores Faltantes", f"{total_missing:,}")
    
    with col4:
        duplicates = df.duplicated().sum()
        st.metric("🔄 Duplicatas", duplicates)
    
    # Análise por coluna
    st.markdown("#### 📊 Análise por Coluna")
    
    quality_data = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        unique_count = df[col].nunique()
        data_type = str(df[col].dtype)
        
        quality_data.append({
            'Coluna': col,
            'Tipo': data_type,
            'Valores Únicos': unique_count,
            'Valores Faltantes': missing_count,
            '% Faltantes': f"{missing_pct:.1f}%",
            'Qualidade': _assess_quality(missing_pct, unique_count, len(df))
        })
    
    quality_df = pd.DataFrame(quality_data)
    st.dataframe(quality_df, use_container_width=True)
    
    # Gráfico de valores faltantes
    if total_missing > 0:
        st.markdown("#### ❌ Valores Faltantes por Coluna")
        
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            fig = px.bar(
                x=missing_data.values,
                y=missing_data.index,
                orientation='h',
                title="Contagem de Valores Faltantes",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_layout(
                xaxis_title="Quantidade de Valores Faltantes",
                yaxis_title="Colunas",
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Outliers (apenas para variáveis numéricas)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        st.markdown("#### 🎯 Detecção de Outliers")
        
        selected_numeric = st.selectbox(
            "Selecione uma variável numérica:",
            numeric_cols
        )
        
        if selected_numeric:
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot para outliers
                fig = px.box(
                    df, y=selected_numeric,
                    title=f"Box Plot: {selected_numeric}",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Estatísticas de outliers
                Q1 = df[selected_numeric].quantile(0.25)
                Q3 = df[selected_numeric].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[selected_numeric] < lower_bound) | 
                             (df[selected_numeric] > upper_bound)]
                
                st.markdown("**📊 Estatísticas de Outliers:**")
                st.write(f"- **Q1 (25%):** {Q1:.2f}")
                st.write(f"- **Q3 (75%):** {Q3:.2f}")
                st.write(f"- **IQR:** {IQR:.2f}")
                st.write(f"- **Limite Inferior:** {lower_bound:.2f}")
                st.write(f"- **Limite Superior:** {upper_bound:.2f}")
                st.write(f"- **Outliers Detectados:** {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

def _assess_quality(missing_pct, unique_count, total_count):
    """Avaliar qualidade da coluna"""
    if missing_pct == 0:
        if unique_count == total_count:
            return "🟢 Excelente (Única)"
        elif unique_count > total_count * 0.8:
            return "🟢 Excelente"
        else:
            return "🟡 Boa"
    elif missing_pct < 5:
        return "🟡 Boa"
    elif missing_pct < 20:
        return "🟠 Moderada"
    else:
        return "🔴 Ruim"