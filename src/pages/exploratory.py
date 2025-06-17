"""
Página de Análise Exploratória
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import os

def show_exploratory_page(data, i18n):
    """
    Mostrar página de análise exploratória
    """
    st.header("📈 Análise Exploratória de Dados")
    
    if data is None or data.empty:
        st.warning("⚠️ Dados não carregados. Verifique a conexão com a fonte de dados.")
        return
    
    # Sidebar para filtros
    st.sidebar.header("🔍 Filtros de Análise")
    
    # Análise univariada vs bivariada
    analysis_type = st.sidebar.selectbox(
        "Tipo de Análise:",
        ["Univariada", "Bivariada", "Correlações", "Outliers"]
    )
    
    if analysis_type == "Univariada":
        show_univariate_analysis(data)
    elif analysis_type == "Bivariada":
        show_bivariate_analysis(data)
    elif analysis_type == "Correlações":
        show_correlation_analysis(data)
    elif analysis_type == "Outliers":
        show_outlier_analysis(data)

def show_univariate_analysis(data):
    """Análise univariada"""
    st.subheader("📊 Análise Univariada")
    
    # Selecionar coluna para análise
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col_type = st.selectbox("Tipo de Variável:", ["Numérica", "Categórica"])
    
    if col_type == "Numérica" and numeric_cols:
        selected_col = st.selectbox("Selecione a coluna:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma
            fig_hist = px.histogram(
                data, 
                x=selected_col,
                title=f"Distribuição de {selected_col}",
                nbins=30
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot
            fig_box = px.box(
                data, 
                y=selected_col,
                title=f"Box Plot - {selected_col}"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Estatísticas
        st.subheader("📋 Estatísticas Descritivas")
        stats_df = pd.DataFrame({
            'Estatística': ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo', 'Assimetria', 'Curtose'],
            'Valor': [
                data[selected_col].mean(),
                data[selected_col].median(),
                data[selected_col].std(),
                data[selected_col].min(),
                data[selected_col].max(),
                data[selected_col].skew(),
                data[selected_col].kurtosis()
            ]
        })
        st.dataframe(stats_df, use_container_width=True)
    
    elif col_type == "Categórica" and categorical_cols:
        selected_col = st.selectbox("Selecione a coluna:", categorical_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de barras
            value_counts = data[selected_col].value_counts().head(15)
            fig_bar = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Frequência - {selected_col}"
            )
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Gráfico de pizza
            fig_pie = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Proporção - {selected_col}"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Tabela de frequências
        st.subheader("📋 Tabela de Frequências")
        freq_df = pd.DataFrame({
            'Categoria': value_counts.index,
            'Frequência': value_counts.values,
            'Percentual': (value_counts.values / len(data) * 100).round(2)
        })
        st.dataframe(freq_df, use_container_width=True)

def show_bivariate_analysis(data):
    """Análise bivariada"""
    st.subheader("📊 Análise Bivariada")
    
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    analysis_mode = st.selectbox(
        "Modo de Análise:",
        ["Numérica vs Numérica", "Numérica vs Categórica", "Categórica vs Categórica"]
    )
    
    if analysis_mode == "Numérica vs Numérica" and len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Variável X:", numeric_cols, key="x_num")
        with col2:
            y_vars = [col for col in numeric_cols if col != x_var]
            y_var = st.selectbox("Variável Y:", y_vars, key="y_num")
        
        # Scatter plot
        fig_scatter = px.scatter(
            data, 
            x=x_var, 
            y=y_var,
            title=f"{y_var} vs {x_var}",
            trendline="ols"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Correlação
        correlation = data[x_var].corr(data[y_var])
        st.metric("Correlação de Pearson", f"{correlation:.3f}")
    
    elif analysis_mode == "Numérica vs Categórica" and numeric_cols and categorical_cols:
        col1, col2 = st.columns(2)
        with col1:
            num_var = st.selectbox("Variável Numérica:", numeric_cols)
        with col2:
            cat_var = st.selectbox("Variável Categórica:", categorical_cols)
        
        # Box plot por categoria
        fig_box = px.box(
            data, 
            x=cat_var, 
            y=num_var,
            title=f"{num_var} por {cat_var}"
        )
        fig_box.update_xaxes(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Estatísticas por grupo
        st.subheader("📊 Estatísticas por Grupo")
        group_stats = data.groupby(cat_var)[num_var].agg(['mean', 'median', 'std', 'count']).round(2)
        st.dataframe(group_stats, use_container_width=True)

def show_correlation_analysis(data):
    """Análise de correlações"""
    st.subheader("🔗 Análise de Correlações")
    
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Necessário pelo menos 2 variáveis numéricas para análise de correlação")
        return
    
    # Matriz de correlação
    corr_matrix = data[numeric_cols].corr()
    
    # Heatmap
    fig_heatmap = px.imshow(
        corr_matrix,
        title="Matriz de Correlação",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Correlações mais fortes
    st.subheader("🔥 Correlações Mais Fortes")
    
    # Extrair correlações (excluindo diagonal)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Variável 1': corr_matrix.columns[i],
                'Variável 2': corr_matrix.columns[j],
                'Correlação': corr_matrix.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(corr_pairs)
    corr_df['Correlação Abs'] = abs(corr_df['Correlação'])
    corr_df = corr_df.sort_values('Correlação Abs', ascending=False)
    
    st.dataframe(corr_df.head(10), use_container_width=True)

def show_outlier_analysis(data):
    """Análise de outliers"""
    st.subheader("🎯 Análise de Outliers")
    
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numeric_cols:
        st.warning("Nenhuma variável numérica encontrada para análise de outliers")
        return
    
    selected_col = st.selectbox("Selecione a variável:", numeric_cols)
    
    # Calcular outliers usando IQR
    Q1 = data[selected_col].quantile(0.25)
    Q3 = data[selected_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[selected_col] < lower_bound) | (data[selected_col] > upper_bound)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total de Outliers", len(outliers))
        st.metric("% de Outliers", f"{len(outliers)/len(data)*100:.2f}%")
    
    with col2:
        st.metric("Limite Inferior", f"{lower_bound:.2f}")
        st.metric("Limite Superior", f"{upper_bound:.2f}")
    
    # Box plot com outliers destacados
    fig_box = px.box(
        data, 
        y=selected_col,
        title=f"Outliers em {selected_col}",
        points="outliers"
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Mostrar outliers
    if len(outliers) > 0:
        st.subheader("📋 Registros com Outliers")
        st.dataframe(outliers, use_container_width=True)

def plot_salary_by_education_hours(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Boxplot: Escolaridade vs Salário
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='education-num', y='hours-per-week', hue='salary', data=df)
    plt.title('Horas de Trabalho por Escolaridade e Faixa Salarial')
    plt.xlabel('Anos de Escolaridade')
    plt.ylabel('Horas por Semana')
    plt.legend(title='Salário')
    plt.tight_layout()
    os.makedirs("output/imagens", exist_ok=True)
    plt.savefig("output/imagens/correlacao_educacao_horas_salario.png", dpi=300, bbox_inches='tight')
    plt.close()  # Opcional: fecha a figura para liberar memória