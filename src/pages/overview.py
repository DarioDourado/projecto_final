"""
📊 Página de Visão Geral
Dashboard principal com métricas e visualizações resumidas
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

def show_overview_page(data, i18n):
    """Página principal de visão geral"""
    from src.components.navigation import show_page_header
    
    # ✅ IMPORT DIRETO DAS FUNÇÕES (correção do erro)
    show_page_header(
        i18n.t('navigation.overview', 'Visão Geral'),
        i18n.t('overview.subtitle', 'Dashboard principal com métricas e insights dos dados'),
        "📊"
    )
    
    df = data.get('df')
    if df is None or len(df) == 0:
        st.warning(i18n.t('messages.pipeline_needed', '⚠️ Execute: python main.py'))
        _show_system_status(data, i18n)
        return
    
    # Status dos dados
    status = data.get('status', '❌ Dados não encontrados')
    if "✅" in status:
        st.success(status)
    elif "⚠️" in status:
        st.warning(status)
    else:
        st.error(status)
    
    # MÉTRICAS PRINCIPAIS - Usando st.metric diretamente (sem create_metric_card)
    _show_main_metrics_corrected(df, i18n)
    
    # Visualizações resumidas
    col1, col2 = st.columns(2)
    
    with col1:
        _show_salary_distribution(df, i18n)
        _show_age_distribution(df, i18n)
    
    with col2:
        _show_education_distribution(df, i18n)
        _show_gender_distribution(df, i18n)
    
    # Métricas avançadas se disponíveis
    _show_advanced_overview(data, i18n)

def _show_main_metrics_corrected(df, i18n):
    """Mostrar métricas principais usando st.metric diretamente"""
    st.markdown(f"## 📋 {i18n.t('overview.main_metrics', 'Métricas Principais')}")
    
    # 4 colunas para as métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Métrica 1: Total de Registros
        total_records = len(df)
        st.metric(
            label=f"📋 {i18n.t('data.records', 'Registros')}",
            value=f"{total_records:,}"
        )
    
    with col2:
        # Métrica 2: Total de Colunas
        total_columns = len(df.columns)
        st.metric(
            label=f"📊 {i18n.t('data.columns', 'Colunas')}",
            value=f"{total_columns}"
        )
    
    with col3:
        # Métrica 3: Taxa de Salário Alto
        if 'salary' in df.columns:
            high_salary_rate = (df['salary'] == '>50K').mean()
            st.metric(
                label=f"💰 {i18n.t('data.high_salary', 'Salário Alto')}",
                value=f"{high_salary_rate:.1%}",
                delta=f"+{high_salary_rate-0.24:.1%}" if high_salary_rate > 0.24 else None
            )
        else:
            st.metric(
                label=f"💰 {i18n.t('data.high_salary', 'Salário Alto')}",
                value="N/A"
            )
    
    with col4:
        # Métrica 4: Taxa de Valores Ausentes - ✅ MÉTRICA EM FALTA ADICIONADA
        missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
        st.metric(
            label=f"❌ {i18n.t('data.missing', 'Missing')}",
            value=f"{missing_rate:.1%}",
            delta=f"-{0.05-missing_rate:.1%}" if missing_rate < 0.05 else None,
            delta_color="inverse"  # Verde para menos missing values
        )
    
    # Métricas secundárias em uma segunda linha
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        # Duplicatas
        if hasattr(df, 'duplicated'):
            duplicates = df.duplicated().sum()
            duplicate_rate = (duplicates / len(df)) * 100
            st.metric(
                label=f"🔄 {i18n.t('data.duplicates', 'Duplicatas')}",
                value=f"{duplicate_rate:.1f}%",
                delta=f"-{2.0-duplicate_rate:.1f}%" if duplicate_rate < 2.0 else None,
                delta_color="inverse"
            )
    
    with col6:
        # Idade média
        if 'age' in df.columns:
            avg_age = df['age'].mean()
            st.metric(
                label=f"🎂 {i18n.t('data.avg_age', 'Idade Média')}",
                value=f"{avg_age:.1f}",
                delta="anos"
            )
    
    with col7:
        # Variáveis numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.metric(
            label=f"🔢 {i18n.t('data.numeric_vars', 'Vars. Numéricas')}",
            value=f"{len(numeric_cols)}"
        )
    
    with col8:
        # Variáveis categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns
        st.metric(
            label=f"📝 {i18n.t('data.categorical_vars', 'Vars. Categóricas')}",
            value=f"{len(categorical_cols)}"
        )

def _show_salary_distribution(df, i18n):
    """Mostrar distribuição de salários"""
    if 'salary' not in df.columns:
        return
    
    st.markdown(f"### 💰 {i18n.t('charts.salary_distribution', 'Distribuição de Salários')}")
    
    salary_counts = df['salary'].value_counts()
    
    # Gráfico de pizza moderno
    fig = go.Figure(data=[go.Pie(
        labels=salary_counts.index,
        values=salary_counts.values,
        hole=0.4,
        marker=dict(
            colors=['#FF6B6B', '#4ECDC4'],
            line=dict(color='#FFFFFF', width=2)
        ),
        textinfo='label+percent',
        textfont_size=12
    )])
    
    fig.update_layout(
        title=i18n.t('charts.salary_distribution', 'Distribuição de Salários'),
        height=400,
        showlegend=True,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _show_age_distribution(df, i18n):
    """Mostrar distribuição de idades"""
    if 'age' not in df.columns:
        return
    
    st.markdown(f"### 🎂 {i18n.t('charts.age_distribution', 'Distribuição de Idades')}")
    
    fig = px.histogram(
        df, 
        x='age',
        nbins=30,
        title=i18n.t('charts.age_distribution', 'Distribuição de Idades'),
        color_discrete_sequence=['#667eea'],
        template="plotly_white"
    )
    
    fig.update_layout(
        xaxis_title=i18n.t('data.age', 'Idade'),
        yaxis_title=i18n.t('charts.frequency', 'Frequência'),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _show_education_distribution(df, i18n):
    """Mostrar distribuição de educação"""
    if 'education' not in df.columns:
        return
    
    st.markdown(f"### 🎓 {i18n.t('charts.education_distribution', 'Distribuição de Educação')}")
    
    # Top 8 níveis de educação
    education_counts = df['education'].value_counts().head(8)
    
    fig = px.bar(
        x=education_counts.values,
        y=education_counts.index,
        orientation='h',
        title=i18n.t('charts.education_distribution', 'Top 8 Níveis de Educação'),
        color=education_counts.values,
        color_continuous_scale='viridis',
        template="plotly_white"
    )
    
    fig.update_layout(
        xaxis_title=i18n.t('charts.count', 'Quantidade'),
        yaxis_title=i18n.t('data.education', 'Educação'),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _show_gender_distribution(df, i18n):
    """Mostrar distribuição por sexo"""
    if 'sex' not in df.columns:
        return
    
    st.markdown(f"### 👥 {i18n.t('charts.sex_distribution', 'Distribuição por Sexo')}")
    
    sex_counts = df['sex'].value_counts()
    
    # Gráfico de barras com cores personalizadas
    colors = ['#FF6B6B', '#4ECDC4']
    
    fig = go.Figure(data=[
        go.Bar(
            x=sex_counts.index,
            y=sex_counts.values,
            marker=dict(
                color=colors[:len(sex_counts)],
                line=dict(color='rgba(50, 50, 50, 0.5)', width=1)
            ),
            text=sex_counts.values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=i18n.t('charts.sex_distribution', 'Distribuição por Sexo'),
        xaxis_title=i18n.t('data.sex', 'Sexo'),
        yaxis_title=i18n.t('charts.count', 'Quantidade'),
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _show_advanced_overview(data, i18n):
    """Mostrar visão geral avançada se dados disponíveis"""
    models = data.get('models', {})
    
    if models:
        st.markdown(f"## 🤖 {i18n.t('overview.models_summary', 'Resumo dos Modelos')}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label=f"🤖 {i18n.t('models.total', 'Total de Modelos')}",
                value=f"{len(models)}"
            )
        
        # Buscar melhor modelo por accuracy
        best_model = None
        best_accuracy = 0
        
        for model_name, model_data in models.items():
            if isinstance(model_data, dict) and 'accuracy' in model_data:
                if model_data['accuracy'] > best_accuracy:
                    best_accuracy = model_data['accuracy']
                    best_model = model_name
        
        if best_model:
            with col2:
                st.metric(
                    label=f"🏆 {i18n.t('models.best_model', 'Melhor Modelo')}",
                    value=best_model
                )
            
            with col3:
                st.metric(
                    label=f"🎯 {i18n.t('models.best_accuracy', 'Melhor Accuracy')}",
                    value=f"{best_accuracy:.3f}",
                    delta=f"+{best_accuracy-0.8:.3f}" if best_accuracy > 0.8 else None
                )
    
    # Status dos arquivos gerados
    _show_files_status(data, i18n)

def _show_files_status(data, i18n):
    """Mostrar status dos arquivos gerados"""
    st.markdown(f"## 📁 {i18n.t('overview.files_status', 'Status dos Arquivos')}")
    
    # Verificar arquivos importantes
    important_paths = {
        'Imagens': Path("output/images"),
        'Análises': Path("output/analysis"), 
        'Modelos': Path("models"),
        'Logs': Path("logs")
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        for i, (name, path) in enumerate(list(important_paths.items())[:2]):
            if path.exists():
                file_count = len(list(path.glob("*")))
                st.metric(f"📂 {name}", file_count)
            else:
                st.metric(f"📂 {name}", "0", "❌ Não encontrado")
    
    with col2:
        for i, (name, path) in enumerate(list(important_paths.items())[2:]):
            if path.exists():
                file_count = len(list(path.glob("*")))
                st.metric(f"📂 {name}", file_count)
            else:
                st.metric(f"📂 {name}", "0", "❌ Não encontrado")

def _show_system_status(data, i18n):
    """Mostrar status do sistema quando não há dados"""
    st.markdown(f"## ⚙️ {i18n.t('overview.system_status', 'Status do Sistema')}")
    
    status_checks = [
        ("📊 Dados CSV", Path("data/raw/4-Carateristicas_salario.csv").exists()),
        ("🗄️ Base de Dados", "sql" in data.get('source', '').lower()),
        ("🤖 Pipeline ML", Path("models").exists()),
        ("📈 Análises", Path("output/analysis").exists()),
        ("🎨 Visualizações", Path("output/images").exists()),
    ]
    
    col1, col2 = st.columns(2)
    
    for i, (check_name, status) in enumerate(status_checks):
        target_col = col1 if i % 2 == 0 else col2
        
        with target_col:
            if status:
                st.success(f"✅ {check_name}")
            else:
                st.error(f"❌ {check_name}")
    
    # Instruções
    st.markdown(f"""
    ### 💡 {i18n.t('overview.instructions', 'Como Começar')}:
    
    1. **{i18n.t('overview.step1', 'Execute o pipeline principal')}**: 
       ```bash
       python main.py
       ```
    
    2. **{i18n.t('overview.step2', 'Aguarde o processamento')}**: O sistema irá carregar dados e treinar modelos
    
    3. **{i18n.t('overview.step3', 'Explore o dashboard')}**: Todas as funcionalidades estarão disponíveis
    """)