"""
Dashboard Streamlit Completo - Análise Salarial Académica VERSÃO FINAL
Sistema interativo com todas as funcionalidades implementadas
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import logging
import warnings
from datetime import datetime
import io

# Configurações
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="Análise Salarial - Dashboard Académico",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS melhorado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        color: #2d3436;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FUNÇÕES AUXILIARES OTIMIZADAS
# =============================================================================

@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_data():
    """Carregar dados com cache otimizado"""
    try:
        # Prioridade: dados processados
        data_paths = [
            Path("data/processed/data_processed.csv"),
            Path("data/raw/4-Carateristicas_salario.csv"),
            Path("bkp/4-Carateristicas_salario.csv"),
            Path("4-Carateristicas_salario.csv")
        ]
        
        for path in data_paths:
            if path.exists():
                df = pd.read_csv(path)
                df = clean_dataframe(df)
                status = "✅ Dados processados" if "processed" in str(path) else "⚠️ Dados brutos"
                return df, f"{status} carregados: {path.name}"
        
        return None, "❌ Nenhum arquivo de dados encontrado!"
    except Exception as e:
        return None, f"❌ Erro ao carregar dados: {e}"

def clean_dataframe(df):
    """Limpeza otimizada do DataFrame"""
    # Converter categóricas
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).replace(['None', 'nan', 'NaN', '?'], 'Unknown')
    
    # Garantir numéricas
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

@st.cache_data
def load_analysis_files():
    """Carregar análises com cache"""
    files_status = {
        'images': [],
        'analysis': [],
        'models': []
    }
    
    # Buscar em múltiplas localizações
    search_paths = {
        'images': [Path("output/images"), Path("imagens"), Path("bkp/imagens")],
        'analysis': [Path("output/analysis"), Path("output"), Path(".")],
        'models': [Path("data/processed"), Path("bkp"), Path(".")]
    }
    
    for category, paths in search_paths.items():
        for path in paths:
            if path.exists():
                if category == 'images':
                    files_status[category].extend(list(path.glob("*.png")))
                elif category == 'analysis':
                    files_status[category].extend(list(path.glob("*.csv")))
                    files_status[category].extend(list(path.glob("*.md")))
                elif category == 'models':
                    files_status[category].extend(list(path.glob("*.joblib")))
    
    # Remover duplicatas
    for category in files_status:
        files_status[category] = list(set(files_status[category]))
    
    return files_status

def create_comparison_chart(df, metric='accuracy'):
    """Criar gráfico de comparação otimizado"""
    if df.empty:
        return None
    
    fig = go.Figure()
    
    # Ordenar por métrica
    df_sorted = df.sort_values(metric, ascending=True)
    
    # Criar barras com gradiente
    colors = px.colors.qualitative.Set3[:len(df_sorted)]
    
    fig.add_trace(go.Bar(
        x=df_sorted[metric],
        y=df_sorted.index,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(50, 50, 50, 0.5)', width=1)
        ),
        text=[f'{x:.3f}' for x in df_sorted[metric]],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f'Comparação de Modelos - {metric.title()}',
        xaxis_title=metric.title(),
        yaxis_title='Modelos',
        height=400,
        showlegend=False
    )
    
    return fig

# =============================================================================
# FUNÇÕES DE GRÁFICOS MODERNIZADOS
# =============================================================================

def create_modern_pie_chart(data, values, names, title, color_scheme='viridis'):
    """Criar gráfico de pizza moderno com efeitos - CORRIGIDO"""
    
    # Verificar se values é array numpy e converter se necessário
    if hasattr(values, 'values'):
        values_list = values.values.tolist()
    elif hasattr(values, 'tolist'):
        values_list = values.tolist()
    else:
        values_list = list(values)
    
    # Verificar se names é array e converter
    if hasattr(names, 'values'):
        names_list = names.values.tolist()
    elif hasattr(names, 'tolist'):
        names_list = names.tolist()
    else:
        names_list = list(names)
    
    # Encontrar o índice do valor máximo de forma segura
    max_index = values_list.index(max(values_list))
    
    fig = go.Figure(data=[go.Pie(
        labels=names_list,
        values=values_list,
        hole=0.4,  # Donut style
        textinfo='label+percent+value',
        textfont_size=12,
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='#FFFFFF', width=2)
        ),
        pull=[0.05 if i == max_index else 0 for i in range(len(values_list))],  # Destacar maior fatia
        hovertemplate='<b>%{label}</b><br>Quantidade: %{value}<br>Percentual: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(size=18, color='#2c3e50')
        ),
        font=dict(family="Arial, sans-serif", size=12),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        ),
        margin=dict(t=60, b=40, l=40, r=120),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_modern_bar_chart(data, x, y, title, color_column=None, orientation='v'):
    """Criar gráfico de barras moderno com gradientes - CORRIGIDO"""
    if orientation == 'v':
        fig = px.bar(
            data, x=x, y=y, 
            color=color_column if color_column else y,
            color_continuous_scale='viridis',
            title=title
        )
        fig.update_traces(
            marker_line_color='rgba(255,255,255,0.8)',
            marker_line_width=2,
            texttemplate='%{y}',
            textposition='outside'
        )
    else:
        fig = px.bar(
            data, x=y, y=x, 
            color=color_column if color_column else y,
            color_continuous_scale='plasma',
            orientation='h',
            title=title
        )
        fig.update_traces(
            marker_line_color='rgba(255,255,255,0.8)',
            marker_line_width=2,
            texttemplate='%{x}',
            textposition='outside'
        )
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(size=18, color='#2c3e50')
        ),
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif"),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
    )
    
    return fig

def create_modern_scatter_plot(data, x, y, color=None, size=None, title="", hover_data=None):
    """Criar scatter plot moderno com animações - CORRIGIDO"""
    
    # Verificar se a coluna size existe e tem valores válidos
    size_col = None
    if size and size in data.columns:
        # Verificar se há valores não-nulos e variação
        size_values = data[size].dropna()
        if len(size_values) > 0 and size_values.var() > 0:
            size_col = size
    
    fig = px.scatter(
        data, x=x, y=y,
        color=color,
        size=size_col,
        hover_data=hover_data,
        color_continuous_scale='viridis',
        title=title,
        opacity=0.7
    )
    
    # Atualizar traces com verificação
    marker_dict = {
        'line': dict(width=1, color='rgba(255,255,255,0.8)')
    }
    
    if size_col:
        max_size = data[size_col].max()
        if max_size > 0:
            marker_dict.update({
                'sizemode': 'diameter',
                'sizeref': 2. * max_size / (40.**2)
            })
    
    fig.update_traces(marker=marker_dict)
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(size=18, color='#2c3e50')
        ),
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif"),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        )
    )
    
    return fig

def create_modern_histogram(data, column, title, bins=30, color_column=None):
    """Criar histograma moderno com densidade - CORRIGIDO"""
    
    # Verificar se a coluna existe
    if column not in data.columns:
        st.error(f"Coluna '{column}' não encontrada no dataset")
        return None
    
    # Verificar se há dados válidos
    valid_data = data[column].dropna()
    if len(valid_data) == 0:
        st.warning(f"Nenhum dado válido encontrado na coluna '{column}'")
        return None
    
    try:
        fig = px.histogram(
            data, x=column,
            nbins=bins,
            color=color_column,
            marginal="box",  # Adicionar boxplot
            title=title,
            opacity=0.7
        )
        
        fig.update_traces(
            marker_line_color='rgba(255,255,255,0.8)',
            marker_line_width=2
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(size=18, color='#2c3e50')
            ),
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif"),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erro ao criar histograma: {e}")
        return None

def create_interactive_correlation_heatmap(df):
    """Criar heatmap de correlação interativo - CORRIGIDO"""
    
    # Selecionar apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        st.warning("Poucas variáveis numéricas para análise de correlação")
        return None
    
    try:
        # Calcular matriz de correlação
        corr_matrix = df[numeric_cols].corr()
        
        # Criar máscara para triângulo superior
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        masked_corr = corr_matrix.where(~mask)
        
        # Preparar dados para o heatmap
        z_data = masked_corr.values
        x_labels = masked_corr.columns.tolist()
        y_labels = masked_corr.index.tolist()
        
        # Criar texto para anotações
        text_data = np.round(z_data, 2).astype(str)
        text_data[mask] = ''  # Remover texto da parte mascarada
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale='RdBu',
            zmid=0,
            text=text_data,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlação: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='<b>🔗 Matriz de Correlação Interativa</b>',
                x=0.5,
                font=dict(size=18, color='#2c3e50')
            ),
            xaxis_title='<b>Variáveis</b>',
            yaxis_title='<b>Variáveis</b>',
            font=dict(family="Arial, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erro ao criar heatmap de correlação: {e}")
        return None

def show_overview_enhanced(df, load_message, files_status):
    """Visão geral com gráficos modernizados - CORRIGIDA"""
    st.header("📊 Visão Geral do Dataset")
    
    # Status message com estilo
    if "processados" in load_message:
        st.success(load_message)
    else:
        st.warning(load_message)
    
    # Métricas principais com cards modernos
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 Registros</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Colunas</h3>
            <h2>{len(df.columns)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'salary' in df.columns:
            high_salary_rate = (df['salary'] == '>50K').mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Salário Alto</h3>
                <h2>{high_salary_rate:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
        st.markdown(f"""
        <div class="metric-card">
            <h3>❌ Missing</h3>
            <h2>{missing_rate:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráficos principais modernizados
    st.subheader("📈 Distribuições Principais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'salary' in df.columns:
            salary_counts = df['salary'].value_counts()
            fig = create_modern_pie_chart(
                data=None,
                values=salary_counts.values,
                names=salary_counts.index,
                title="💰 Distribuição de Salário"
            )
            if fig:  # Verificar se o gráfico foi criado com sucesso
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'sex' in df.columns:
            sex_data = df['sex'].value_counts().reset_index()
            sex_data.columns = ['sex', 'count']
            fig = create_modern_bar_chart(
                data=sex_data,
                x='sex',
                y='count',
                title="👥 Distribuição por Sexo"
            )
            if fig:  # Verificar se o gráfico foi criado com sucesso
                st.plotly_chart(fig, use_container_width=True)
    
    # Gráficos adicionais com verificação de erro
    st.subheader("📈 Análises Detalhadas")
    
    tab1, tab2, tab3 = st.tabs(["📊 Distribuições", "🔗 Correlações", "📋 Estatísticas"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'age' in df.columns:
                fig = create_modern_histogram(
                    data=df,
                    column='age',
                    title="📊 Distribuição de Idade",
                    color_column='salary' if 'salary' in df.columns else None
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'education-num' in df.columns:
                fig = create_modern_histogram(
                    data=df,
                    column='education-num',
                    title="🎓 Anos de Educação",
                    color_column='salary' if 'salary' in df.columns else None
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            fig = create_interactive_correlation_heatmap(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Poucas variáveis numéricas para análise de correlação")
    
    with tab3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
        else:
            st.info("Nenhuma variável numérica encontrada")

# =============================================================================
# FUNÇÕES PRINCIPAIS MODIFICADAS
# =============================================================================

# Session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "📊 Visão Geral"
if 'filters' not in st.session_state:
    st.session_state.filters = {}

def main():
    """Interface principal otimizada"""
    
    # Header elegante
    st.markdown('<div class="main-header">💰 Dashboard de Análise Salarial</div>', 
                unsafe_allow_html=True)
    
    # Carregar dados
    df, load_message = load_data()
    files_status = load_analysis_files()
    
    # Sidebar melhorada
    with st.sidebar:
        st.markdown("## 🎛️ Controle Central")
        
        # Status do sistema com cores
        pipeline_executed = len(files_status['models']) > 0
        if pipeline_executed:
            st.markdown('<div class="success-box">✅ Pipeline Executado!</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ Execute: python main.py</div>', 
                       unsafe_allow_html=True)
        
        # Métricas do sistema
        st.markdown("### 📊 Arquivos Encontrados")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎨 Imagens", len(files_status['images']))
            st.metric("📊 Análises", len(files_status['analysis']))
        with col2:
            st.metric("🤖 Modelos", len(files_status['models']))
            if df is not None:
                st.metric("📋 Registros", f"{len(df):,}")
        
        # Navegação principal
        st.markdown("### 🧭 Navegação")
        pages = [
            ("📊 Visão Geral", "overview"),
            ("📈 Análise Exploratória", "exploratory"),
            ("🤖 Modelos ML", "models"),
            ("🎯 Clustering", "clustering"),
            ("📋 Regras de Associação", "rules"),
            ("📊 Métricas Avançadas", "metrics"),
            ("🔮 Predição", "prediction"),
            ("📁 Relatórios", "reports")
        ]
        
        for page_name, page_key in pages:
            if st.button(page_name, key=f"nav_{page_key}"):
                st.session_state.current_page = page_name
        
        # Filtros se dados disponíveis
        if df is not None and len(df) > 0:
            st.markdown("### 🔍 Filtros")
            
            # Reset filters button
            if st.button("🗑️ Limpar Filtros"):
                st.session_state.filters = {}
                st.rerun()
            
            # Salary filter
            if 'salary' in df.columns:
                salary_options = [str(x) for x in df['salary'].unique() if str(x) != 'Unknown']
                if salary_options:
                    salary_filter = st.multiselect("💰 Salário", salary_options)
                    if salary_filter:
                        st.session_state.filters['salary'] = salary_filter
            
            # Age filter
            if 'age' in df.columns:
                age_range = st.slider("🎂 Idade", 
                                    int(df['age'].min()), 
                                    int(df['age'].max()),
                                    (int(df['age'].min()), int(df['age'].max())))
                if age_range != (int(df['age'].min()), int(df['age'].max())):
                    st.session_state.filters['age'] = age_range
        
        # Status de filtros
        if st.session_state.filters:
            st.markdown("### 📋 Filtros Ativos")
            for filter_name, filter_value in st.session_state.filters.items():
                if isinstance(filter_value, list):
                    st.write(f"• **{filter_name}**: {', '.join(map(str, filter_value))}")
                elif isinstance(filter_value, tuple):
                    st.write(f"• **{filter_name}**: {filter_value[0]} - {filter_value[1]}")
    
    # Aplicar filtros
    if df is not None:
        filtered_df = apply_filters(df, st.session_state.filters)
        if len(filtered_df) != len(df):
            st.info(f"🔍 Filtros: {len(filtered_df):,} de {len(df):,} registros ({len(filtered_df)/len(df):.1%})")
    else:
        filtered_df = None
    
    # Verificação de dados
    if filtered_df is None or len(filtered_df) == 0:
        st.error("❌ Nenhum dado disponível após filtros")
        st.info("Verifique os filtros ou execute: `python main.py`")
        return
    
    # Roteamento de páginas
    current_page = st.session_state.current_page
    
    if current_page == "📊 Visão Geral":
        show_overview_enhanced(filtered_df, load_message, files_status)
    elif current_page == "📈 Análise Exploratória":
        show_exploratory_analysis_enhanced(filtered_df)
    elif current_page == "🤖 Modelos ML":
        show_ml_models_enhanced(filtered_df, files_status)
    elif current_page == "🎯 Clustering":
        show_clustering_analysis_enhanced(filtered_df, files_status)
    elif current_page == "📋 Regras de Associação":
        show_association_rules_enhanced(filtered_df, files_status)
    elif current_page == "📊 Métricas Avançadas":
        show_advanced_metrics_enhanced(filtered_df, files_status)
    elif current_page == "🔮 Predição":
        show_prediction_interface_enhanced(filtered_df, files_status)
    elif current_page == "📁 Relatórios":
        show_reports_enhanced(files_status)

def apply_filters(df, filters):
    """Aplicar filtros de forma otimizada"""
    filtered_df = df.copy()
    
    for col, values in filters.items():
        if col in df.columns:
            if isinstance(values, list) and values:
                filtered_df = filtered_df[filtered_df[col].isin(values)]
            elif isinstance(values, tuple) and len(values) == 2:
                filtered_df = filtered_df[
                    (filtered_df[col] >= values[0]) & 
                    (filtered_df[col] <= values[1])
                ]
    
    return filtered_df

# =============================================================================
# PÁGINAS IMPLEMENTADAS COMPLETAMENTE
# =============================================================================

def show_overview_enhanced(df, load_message, files_status):
    """Visão geral com gráficos modernizados - CORRIGIDA"""
    st.header("📊 Visão Geral do Dataset")
    
    # Status message com estilo
    if "processados" in load_message:
        st.success(load_message)
    else:
        st.warning(load_message)
    
    # Métricas principais com cards modernos
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 Registros</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Colunas</h3>
            <h2>{len(df.columns)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'salary' in df.columns:
            high_salary_rate = (df['salary'] == '>50K').mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Salário Alto</h3>
                <h2>{high_salary_rate:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
        st.markdown(f"""
        <div class="metric-card">
            <h3>❌ Missing</h3>
            <h2>{missing_rate:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráficos principais modernizados
    st.subheader("📈 Distribuições Principais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'salary' in df.columns:
            salary_counts = df['salary'].value_counts()
            fig = create_modern_pie_chart(
                data=None,
                values=salary_counts.values,
                names=salary_counts.index,
                title="💰 Distribuição de Salário"
            )
            if fig:  # Verificar se o gráfico foi criado com sucesso
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'sex' in df.columns:
            sex_data = df['sex'].value_counts().reset_index()
            sex_data.columns = ['sex', 'count']
            fig = create_modern_bar_chart(
                data=sex_data,
                x='sex',
                y='count',
                title="👥 Distribuição por Sexo"
            )
            if fig:  # Verificar se o gráfico foi criado com sucesso
                st.plotly_chart(fig, use_container_width=True)
    
    # Gráficos adicionais com verificação de erro
    st.subheader("📈 Análises Detalhadas")
    
    tab1, tab2, tab3 = st.tabs(["📊 Distribuições", "🔗 Correlações", "📋 Estatísticas"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'age' in df.columns:
                fig = create_modern_histogram(
                    data=df,
                    column='age',
                    title="📊 Distribuição de Idade",
                    color_column='salary' if 'salary' in df.columns else None
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'education-num' in df.columns:
                fig = create_modern_histogram(
                    data=df,
                    column='education-num',
                    title="🎓 Anos de Educação",
                    color_column='salary' if 'salary' in df.columns else None
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            fig = create_interactive_correlation_heatmap(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Poucas variáveis numéricas para análise de correlação")
    
    with tab3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
        else:
            st.info("Nenhuma variável numérica encontrada")

def show_exploratory_analysis_enhanced(df):
    """Análise exploratória com gráficos modernizados - CORRIGIDA"""
    st.header("📈 Análise Exploratória Avançada")
    
    # Controles modernizados
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                x_var = st.selectbox("🔢 Variável X:", numeric_cols)
        
        with col2:
            y_var = st.selectbox("📊 Variável Y:", ["Nenhuma"] + numeric_cols)
        
        with col3:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                color_var = st.selectbox("🎨 Cor por:", ["Nenhuma"] + categorical_cols)
    
    # Gráficos baseados na seleção
    if 'x_var' in locals():
        if y_var != "Nenhuma":
            # Scatter plot
            fig = create_modern_scatter_plot(
                data=df,
                x=x_var,
                y=y_var,
                color=color_var if color_var != "Nenhuma" else None,
                title=f"📊 {x_var} vs {y_var}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Histograma
            fig = create_modern_histogram(
                data=df,
                column=x_var,
                title=f"📊 Distribuição de {x_var}",
                color_column=color_var if color_var != "Nenhuma" else None
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Análises por categoria
    st.subheader("🎯 Análises por Categoria")
    
    if 'salary' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'workclass' in df.columns:
                workclass_salary = df.groupby('workclass')['salary'].apply(
                    lambda x: (x == '>50K').mean()
                ).reset_index()
                workclass_salary.columns = ['workclass', 'high_salary_rate']
                
                fig = create_modern_bar_chart(
                    data=workclass_salary,
                    x='workclass',
                    y='high_salary_rate',
                    title="💼 Taxa de Salário Alto por Classe Trabalhadora"
                )
                # CORREÇÃO: Usar update_layout ao invés de update_xaxis
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'marital-status' in df.columns:
                marital_salary = df.groupby('marital-status')['salary'].apply(
                    lambda x: (x == '>50K').mean()
                ).reset_index()
                marital_salary.columns = ['marital_status', 'high_salary_rate']
                
                fig = create_modern_bar_chart(
                    data=marital_salary,
                    x='marital_status',
                    y='high_salary_rate',
                    title="💑 Taxa de Salário Alto por Estado Civil"
                )
                # CORREÇÃO: Usar update_layout ao invés de update_xaxis
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

def show_ml_models_enhanced(df, files_status):
    """Modelos ML implementados"""
    st.header("🤖 Modelos de Machine Learning")
    
    # Verificar se há modelos
    if not files_status['models']:
        st.warning("⚠️ Nenhum modelo encontrado. Execute: python main.py")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Comparação", "🎯 Features", "📈 Performance"])
    
    with tab1:
        # Procurar arquivo de comparação
        comparison_file = None
        for file in files_status['analysis']:
            if 'model_comparison' in file.name or 'comparison' in file.name:
                comparison_file = file
                break
        
        if comparison_file:
            try:
                comparison_df = pd.read_csv(comparison_file)
                if 'model_name' in comparison_df.columns:
                    comparison_df.set_index('model_name', inplace=True)
                
                st.dataframe(comparison_df.round(4), use_container_width=True)
                
                # Gráfico de comparação
                if 'accuracy' in comparison_df.columns:
                    fig = create_comparison_chart(comparison_df, 'accuracy')
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erro ao carregar comparação: {e}")
        else:
            st.info("Execute o pipeline para gerar comparação de modelos")
    
    with tab2:
        # Feature importance
        importance_files = [f for f in files_status['analysis'] 
                          if 'importance' in f.name or 'feature' in f.name]
        
        if importance_files:
            selected_file = st.selectbox("Modelo:", 
                                       [f.stem for f in importance_files])
            
            # Carregar arquivo selecionado
            for file in importance_files:
                if selected_file in file.stem:
                    try:
                        importance_df = pd.read_csv(file)
                        if 'feature' in importance_df.columns and 'importance' in importance_df.columns:
                            top_features = importance_df.head(15)
                            
                            fig = px.bar(top_features, x='importance', y='feature',
                                       orientation='h', title="Feature Importance")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.dataframe(importance_df.head(20), use_container_width=True)
                    except Exception as e:
                        st.error(f"Erro: {e}")
                    break
        else:
            st.info("Execute o pipeline para gerar feature importance")
    
    with tab3:
        # Imagens de performance
        performance_images = [f for f in files_status['images'] 
                            if any(keyword in f.name.lower() 
                                  for keyword in ['roc', 'curve', 'performance'])]
        
        if performance_images:
            for img in performance_images:
                st.image(str(img), caption=img.name, use_column_width=True)
        else:
            st.info("Execute o pipeline para gerar gráficos de performance")

def show_clustering_analysis_enhanced(df, files_status):
    """Análise de clustering com visualizações modernizadas"""
    st.header("🎯 Análise de Clustering Modernizada")
    
    # Verificar arquivos existentes
    clustering_files = {
        'analysis_image': Path("output/images/clustering_analysis.png"),
        'pca_image': Path("output/images/clusters_pca_visualization.png"),
        'cluster_data': Path("output/analysis/clustering_results.csv")
    }
    
    files_found = {name: path.exists() for name, path in clustering_files.items()}
    
    # Dashboard de status
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "✅" if files_found['analysis_image'] else "❌"
        st.markdown(f"""
        <div class="{'success-box' if files_found['analysis_image'] else 'warning-box'}">
            <h4>{status} Análise do Cotovelo</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = "✅" if files_found['pca_image'] else "❌"
        st.markdown(f"""
        <div class="{'success-box' if files_found['pca_image'] else 'warning-box'}">
            <h4>{status} Visualização PCA</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = "✅" if files_found['cluster_data'] else "❌"
        st.markdown(f"""
        <div class="{'success-box' if files_found['cluster_data'] else 'warning-box'}">
            <h4>{status} Dados de Cluster</h4>
        </div>
        """, unsafe_allow_html=True)
    
    # Clustering interativo modernizado
    st.subheader("🔧 Clustering Interativo Avançado")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            algorithm = st.selectbox("🤖 Algoritmo", ["K-Means", "DBSCAN", "Agglomerative"])
        
        with col2:
            n_clusters = st.slider("📊 Número de Clusters", 2, 10, 4)
        
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'salary' in numeric_cols:
                numeric_cols.remove('salary')
            
            features = st.multiselect(
                "🎯 Features",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
    
    if len(features) >= 2 and st.button("🚀 Executar Clustering Avançado", type="primary"):
        try:
            from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
            
            with st.spinner("🔄 Processando clustering..."):
                # Preparar dados
                X = df[features].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Aplicar algoritmo selecionado
                if algorithm == "K-Means":
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                elif algorithm == "DBSCAN":
                    clusterer = DBSCAN(eps=0.5, min_samples=5)
                else:  # Agglomerative
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                
                clusters = clusterer.fit_predict(X_scaled)
                
                # Calcular métricas
                if len(set(clusters)) > 1:
                    silhouette = silhouette_score(X_scaled, clusters)
                else:
                    silhouette = 0
                
                # Visualizações modernas
                col1, col2 = st.columns(2)
                
                with col1:
                    # Scatter plot 2D
                    if len(features) >= 2:
                        plot_df = X.copy()
                        plot_df['Cluster'] = [f'Cluster {i}' for i in clusters]
                        
                        fig = create_modern_scatter_plot(
                            data=plot_df,
                            x=features[0],
                            y=features[1],
                            color='Cluster',
                            title=f"🎯 Clustering {algorithm}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # PCA Visualization
                    if len(features) > 2:
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        pca_df = pd.DataFrame({
                            'PC1': X_pca[:, 0],
                            'PC2': X_pca[:, 1],
                            'Cluster': [f'Cluster {i}' for i in clusters]
                        })
                        
                        fig = create_modern_scatter_plot(
                            data=pca_df,
                            x='PC1',
                            y='PC2',
                            color='Cluster',
                            title="🎨 Visualização PCA"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Estatísticas dos clusters
                unique_clusters, counts = np.unique(clusters, return_counts=True)
                cluster_stats = pd.DataFrame({
                    'Cluster': [f'Cluster {i}' for i in unique_clusters],
                    'Tamanho': counts,
                    'Percentual': (counts / len(clusters) * 100).round(1)
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Estatísticas dos Clusters")
                    st.dataframe(cluster_stats, use_container_width=True)
                
                with col2:
                    # Gráfico de pizza dos clusters
                    fig = create_modern_pie_chart(
                        data=None,
                        values=counts,
                        names=[f'Cluster {i}' for i in unique_clusters],
                        title="📊 Distribuição dos Clusters"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Métricas de qualidade
                st.markdown("### 📈 Métricas de Qualidade")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Silhouette Score", f"{silhouette:.3f}")
                
                with col2:
                    st.metric("📊 Número de Clusters", len(unique_clusters))
                
                with col3:
                    st.metric("📋 Pontos Processados", len(X))
                
                st.success(f"✅ Clustering {algorithm} executado com sucesso!")
        
        except Exception as e:
            st.error(f"❌ Erro no clustering: {e}")
    
    # Mostrar imagens existentes se disponíveis
    if any(files_found.values()):
        st.subheader("📊 Análises Pré-computadas")
        
        for name, path in clustering_files.items():
            if path.exists() and path.suffix == '.png':
                st.image(str(path), caption=path.name, use_column_width=True)

def show_association_rules_enhanced(df, files_status):
    """Regras de associação implementadas"""
    st.header("📋 Regras de Associação")
    
    # Procurar arquivo de regras
    rules_files = [f for f in files_status['analysis'] 
                  if 'rule' in f.name.lower() or 'association' in f.name.lower()]
    
    if rules_files:
        for file in rules_files:
            try:
                rules_df = pd.read_csv(file)
                st.dataframe(rules_df, use_container_width=True)
                
                # Gráfico de suporte vs confiança
                if 'support' in rules_df.columns and 'confidence' in rules_df.columns:
                    fig = px.scatter(rules_df, x='support', y='confidence',
                                   hover_data=['antecedents', 'consequents'] if 'antecedents' in rules_df.columns else None,
                                   title="Suporte vs Confiança das Regras")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erro ao carregar regras: {e}")
    else:
        st.info("Execute o pipeline para gerar regras de associação")
        
        # Análise básica de co-ocorrência
        st.subheader("📊 Análise Básica de Co-ocorrência")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                var1 = st.selectbox("Variável 1:", categorical_cols)
            with col2:
                var2 = st.selectbox("Variável 2:", 
                                  [c for c in categorical_cols if c != var1])
            
            if st.button("🔍 Analisar Co-ocorrência"):
                crosstab = pd.crosstab(df[var1], df[var2])
                st.dataframe(crosstab, use_container_width=True)
                
                # Heatmap
                fig = px.imshow(crosstab.values, 
                              x=crosstab.columns, y=crosstab.index,
                              title=f"Co-ocorrência: {var1} vs {var2}")
                st.plotly_chart(fig, use_container_width=True)

def show_advanced_metrics_enhanced(df, files_status):
    """Métricas avançadas implementadas"""
    st.header("📊 Métricas Avançadas")
    
    # Procurar relatórios de métricas
    metrics_files = [f for f in files_status['analysis'] 
                    if 'metric' in f.name.lower() or 'advanced' in f.name.lower()]
    
    if metrics_files:
        for file in metrics_files:
            try:
                metrics_df = pd.read_csv(file)
                st.dataframe(metrics_df, use_container_width=True)
            except Exception as e:
                st.error(f"Erro: {e}")
    else:
        st.info("Execute o pipeline para gerar métricas avançadas")
    
    # Métricas calculadas em tempo real
    st.subheader("⚡ Métricas em Tempo Real")
    
    if 'salary' in df.columns:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Taxa de conversão por educação
            if 'education' in df.columns:
                edu_conversion = df.groupby('education')['salary'].apply(
                    lambda x: (x == '>50K').mean()
                ).sort_values(ascending=False)
                
                st.write("**Taxa Salário Alto por Educação:**")
                for edu, rate in edu_conversion.head(5).items():
                    st.write(f"• {edu}: {rate:.1%}")
        
        with col2:
            # Taxa por idade
            if 'age' in df.columns:
                age_bins = [0, 25, 35, 45, 55, 100]
                age_labels = ['18-25', '26-35', '36-45', '46-55', '55+']
                df_temp = df.copy()
                df_temp['age_group'] = pd.cut(df_temp['age'], bins=age_bins, 
                                            labels=age_labels, include_lowest=True)
                
                age_conversion = df_temp.groupby('age_group')['salary'].apply(
                    lambda x: (x == '>50K').mean()
                )
                
                st.write("**Taxa por Faixa Etária:**")
                for age_group, rate in age_conversion.items():
                    if not pd.isna(rate):
                        st.write(f"• {age_group}: {rate:.1%}")
        
        with col3:
            # Taxa por sexo
            if 'sex' in df.columns:
                sex_conversion = df.groupby('sex')['salary'].apply(
                    lambda x: (x == '>50K').mean()
                )
                
                st.write("**Taxa por Sexo:**")
                for sex, rate in sex_conversion.items():
                    st.write(f"• {sex}: {rate:.1%}")

def show_prediction_interface_enhanced(df, files_status):
    """Interface de predição implementada"""
    st.header("🔮 Predição Interativa")
    
    # Verificar se há modelos
    if not files_status['models']:
        st.warning("⚠️ Nenhum modelo encontrado para predição")
        return
    
    # Tentar carregar um modelo
    model = None
    for model_file in files_status['models']:
        try:
            model = joblib.load(model_file)
            st.success(f"✅ Modelo carregado: {model_file.name}")
            break
        except Exception:
            continue
    
    if model is None:
        st.error("❌ Não foi possível carregar nenhum modelo")
        return
    
    # Interface de predição
    st.subheader("🎯 Fazer Predição")
    
    # Campos de entrada baseados no dataset
    col1, col2 = st.columns(2)
    
    with col1:
        if 'age' in df.columns:
            age = st.slider("Idade", 
                          int(df['age'].min()), 
                          int(df['age'].max()), 
                          int(df['age'].mean()))
        
        if 'education-num' in df.columns:
            education_num = st.slider("Anos de Educação",
                                    int(df['education-num'].min()),
                                    int(df['education-num'].max()),
                                    int(df['education-num'].mean()))
    
    with col2:
        if 'hours-per-week' in df.columns:
            hours = st.slider("Horas por Semana",
                            int(df['hours-per-week'].min()),
                            int(df['hours-per-week'].max()),
                            int(df['hours-per-week'].mean()))
        
        if 'sex' in df.columns:
            sex = st.selectbox("Sexo", df['sex'].unique())
    
    # Botão de predição
    if st.button("🚀 Fazer Predição"):
        st.info("Interface de predição em desenvolvimento. Valores configurados mas modelo precisa ser adaptado.")

def show_reports_enhanced(files_status):
    """Relatórios implementados"""
    st.header("📁 Relatórios Gerados")
    
    # Mostrar arquivos de relatório
    report_files = [f for f in files_status['analysis'] 
                   if f.suffix in ['.md', '.txt', '.csv']]
    
    if report_files:
        for file in report_files:
            with st.expander(f"📄 {file.name}"):
                try:
                    if file.suffix == '.md':
                        content = file.read_text(encoding='utf-8')
                        st.markdown(content)
                    elif file.suffix == '.csv':
                        df = pd.read_csv(file)
                        st.dataframe(df, use_container_width=True)
                    else:
                        content = file.read_text(encoding='utf-8')
                        st.text(content)
                except Exception as e:
                    st.error(f"Erro ao ler arquivo: {e}")
    else:
        st.info("Execute o pipeline para gerar relatórios")
    
    # Download de relatórios
    st.subheader("📥 Downloads")
    
    if files_status['analysis']:
        st.write("**Arquivos de Análise Disponíveis:**")
        for file in files_status['analysis']:
            if file.suffix == '.csv':
                try:
                    df = pd.read_csv(file)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=f"⬇️ {file.name}",
                        data=csv,
                        file_name=file.name,
                        mime='text/csv'
                    )
                except Exception:
                    continue

# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    main()