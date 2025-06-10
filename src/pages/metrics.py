"""
📊 Página de Métricas Avançadas
Dashboard completo de métricas e KPIs do sistema
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

def show_metrics_page(data, i18n):
    """Página principal de métricas avançadas"""
    from src.components.navigation import show_page_header
    
    show_page_header(
        i18n.t('navigation.advanced_metrics', 'Métricas Avançadas'),
        i18n.t('metrics.subtitle', 'Dashboard completo de KPIs e métricas do sistema'),
        "📊"
    )
    
    df = data.get('df')
    if df is None or len(df) == 0:
        st.warning(i18n.t('messages.pipeline_needed', 'Execute pipeline primeiro'))
        return
    
    # Tabs para diferentes tipos de métricas
    tab1, tab2, tab3, tab4 = st.tabs([
        f"📈 {i18n.t('metrics.data_quality', 'Qualidade dos Dados')}",
        f"🤖 {i18n.t('metrics.model_performance', 'Performance dos Modelos')}",
        f"📊 {i18n.t('metrics.business', 'Métricas de Negócio')}",
        f"🔧 {i18n.t('metrics.system', 'Métricas do Sistema')}"
    ])
    
    with tab1:
        _show_data_quality_metrics(df, i18n)
    
    with tab2:
        _show_model_metrics(data, i18n)
    
    with tab3:
        _show_business_metrics(df, i18n)
    
    with tab4:
        _show_system_metrics(i18n)

def _show_data_quality_metrics(df, i18n):
    """Métricas de qualidade dos dados"""
    st.subheader(f"📋 {i18n.t('metrics.data_quality_title', 'Qualidade dos Dados')}")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric(
            f"{i18n.t('metrics.completeness', 'Completude')}", 
            f"{completeness:.1f}%",
            delta=f"+{completeness-80:.1f}%" if completeness > 80 else None
        )
    
    with col2:
        duplicates = df.duplicated().sum()
        duplicate_rate = (duplicates / len(df)) * 100
        st.metric(
            f"{i18n.t('metrics.duplicates', 'Duplicatas')}", 
            f"{duplicate_rate:.1f}%",
            delta=f"-{100-duplicate_rate:.1f}%" if duplicate_rate < 5 else None
        )
    
    with col3:
        consistency = _calculate_consistency_score(df)
        st.metric(
            f"{i18n.t('metrics.consistency', 'Consistência')}", 
            f"{consistency:.1f}%"
        )
    
    with col4:
        validity = _calculate_validity_score(df)
        st.metric(
            f"{i18n.t('metrics.validity', 'Validade')}", 
            f"{validity:.1f}%"
        )
    
    # Gráfico de missing values por coluna
    missing_by_column = df.isnull().sum()
    if missing_by_column.sum() > 0:
        fig = px.bar(
            x=missing_by_column.index,
            y=missing_by_column.values,
            title=f"📊 {i18n.t('metrics.missing_by_column', 'Valores Ausentes por Coluna')}",
            template="plotly_white"
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def _show_model_metrics(data, i18n):
    """Métricas de performance dos modelos"""
    st.subheader(f"🤖 {i18n.t('metrics.model_performance_title', 'Performance dos Modelos')}")
    
    models = data.get('models', {})
    
    if not models:
        st.info(i18n.t('metrics.no_models', 'Nenhum modelo encontrado. Execute o pipeline primeiro.'))
        return
    
    # Comparação de modelos
    model_comparison = []
    for model_name, model_data in models.items():
        if isinstance(model_data, dict):
            model_comparison.append({
                'Modelo': model_name,
                'Accuracy': model_data.get('accuracy', 0),
                'Precision': model_data.get('precision', 0),
                'Recall': model_data.get('recall', 0),
                'F1-Score': model_data.get('f1', 0),
                'Tempo (s)': model_data.get('training_time', 0)
            })
    
    if model_comparison:
        comparison_df = pd.DataFrame(model_comparison)
        
        # Métricas de destaque
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
            st.metric(
                f"🎯 {i18n.t('metrics.best_accuracy', 'Melhor Accuracy')}",
                f"{best_accuracy['Accuracy']:.3f}",
                best_accuracy['Modelo']
            )
        
        with col2:
            best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
            st.metric(
                f"⚖️ {i18n.t('metrics.best_f1', 'Melhor F1-Score')}",
                f"{best_f1['F1-Score']:.3f}",
                best_f1['Modelo']
            )
        
        with col3:
            fastest = comparison_df.loc[comparison_df['Tempo (s)'].idxmin()]
            st.metric(
                f"⚡ {i18n.t('metrics.fastest', 'Mais Rápido')}",
                f"{fastest['Tempo (s)']:.2f}s",
                fastest['Modelo']
            )
        
        # Tabela de comparação
        st.dataframe(comparison_df, use_container_width=True)
        
        # Gráfico radar dos modelos
        _create_radar_chart(comparison_df, i18n)

def _show_business_metrics(df, i18n):
    """Métricas de negócio"""
    st.subheader(f"💼 {i18n.t('metrics.business_title', 'Métricas de Negócio')}")
    
    if 'salary' not in df.columns:
        st.warning(i18n.t('metrics.no_salary_column', 'Coluna salary não encontrada'))
        return
    
    # KPIs principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_salary_rate = (df['salary'] == '>50K').mean() * 100
        st.metric(
            f"💰 {i18n.t('metrics.high_salary_rate', 'Taxa Salário Alto')}",
            f"{high_salary_rate:.1f}%"
        )
    
    with col2:
        if 'education' in df.columns:
            high_ed_high_salary = df[df['education'].isin(['Bachelors', 'Masters', 'Doctorate'])]
            if len(high_ed_high_salary) > 0:
                ed_impact = (high_ed_high_salary['salary'] == '>50K').mean() * 100
                st.metric(
                    f"🎓 {i18n.t('metrics.education_impact', 'Impacto Educação')}",
                    f"{ed_impact:.1f}%"
                )
    
    with col3:
        if 'sex' in df.columns:
            gender_gap = (
                (df[df['sex'] == 'Male']['salary'] == '>50K').mean() -
                (df[df['sex'] == 'Female']['salary'] == '>50K').mean()
            ) * 100
            st.metric(
                f"⚖️ {i18n.t('metrics.gender_gap', 'Gap de Gênero')}",
                f"{gender_gap:.1f}pp"
            )
    
    with col4:
        if 'age' in df.columns:
            avg_age_high_salary = df[df['salary'] == '>50K']['age'].mean()
            st.metric(
                f"🎂 {i18n.t('metrics.avg_age_high', 'Idade Média (>50K)')}",
                f"{avg_age_high_salary:.1f}"
            )
    
    # Análise por segmentos
    _show_segment_analysis(df, i18n)

def _show_system_metrics(i18n):
    """Métricas do sistema"""
    st.subheader(f"🔧 {i18n.t('metrics.system_title', 'Métricas do Sistema')}")
    
    # Verificar arquivos do sistema
    system_paths = {
        'Models': Path("models"),
        'Images': Path("output/images"),
        'Analysis': Path("output/analysis"),
        'Logs': Path("logs")
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 📁 {i18n.t('metrics.file_status', 'Status dos Arquivos')}")
        for name, path in system_paths.items():
            if path.exists():
                file_count = len(list(path.glob("*")))
                st.metric(f"📂 {name}", file_count)
            else:
                st.metric(f"📂 {name}", "0", "❌ Não encontrado")
    
    with col2:
        st.markdown(f"### 💾 {i18n.t('metrics.storage', 'Armazenamento')}")
        
        total_size = 0
        for path in system_paths.values():
            if path.exists():
                for file in path.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size
        
        # Converter para MB
        total_size_mb = total_size / (1024 * 1024)
        st.metric(f"💽 {i18n.t('metrics.total_size', 'Tamanho Total')}", f"{total_size_mb:.1f} MB")
        
        # Processo de memória (se disponível)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            st.metric(f"🧠 {i18n.t('metrics.memory_usage', 'Uso de Memória')}", f"{memory_mb:.1f} MB")
        except ImportError:
            st.info(i18n.t('metrics.install_psutil', 'Instale psutil para métricas de memória'))

def _calculate_consistency_score(df):
    """Calcular score de consistência dos dados"""
    # Implementação simplificada
    consistency_checks = []
    
    # Verificar ranges válidos para idade
    if 'age' in df.columns:
        valid_age = ((df['age'] >= 18) & (df['age'] <= 100)).mean()
        consistency_checks.append(valid_age)
    
    # Verificar consistência entre educação e anos de educação
    if 'education' in df.columns and 'education_years' in df.columns:
        # Esta é uma verificação simplificada
        education_consistency = 0.9  # Placeholder
        consistency_checks.append(education_consistency)
    
    return np.mean(consistency_checks) * 100 if consistency_checks else 95

def _calculate_validity_score(df):
    """Calcular score de validade dos dados"""
    # Verificações de formato e tipo
    validity_checks = []
    
    # Verificar se colunas categóricas têm valores esperados
    if 'sex' in df.columns:
        valid_sex = df['sex'].isin(['Male', 'Female', 'M', 'F']).mean()
        validity_checks.append(valid_sex)
    
    if 'salary' in df.columns:
        valid_salary = df['salary'].isin(['>50K', '<=50K']).mean()
        validity_checks.append(valid_salary)
    
    return np.mean(validity_checks) * 100 if validity_checks else 92

def _create_radar_chart(comparison_df, i18n):
    """Criar gráfico radar para comparação de modelos"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    for _, row in comparison_df.iterrows():
        values = [row[metric] for metric in metrics]
        values.append(values[0])  # Fechar o polígono
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=row['Modelo'],
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title=f"📊 {i18n.t('metrics.model_radar', 'Comparação Radar dos Modelos')}"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _show_segment_analysis(df, i18n):
    """Análise por segmentos"""
    st.markdown(f"### 🎯 {i18n.t('metrics.segment_analysis', 'Análise por Segmentos')}")
    
    if 'workclass' in df.columns and 'salary' in df.columns:
        # Análise por classe trabalhadora
        workclass_analysis = df.groupby('workclass')['salary'].apply(
            lambda x: (x == '>50K').mean()
        ).sort_values(ascending=False)
        
        fig = px.bar(
            x=workclass_analysis.index,
            y=workclass_analysis.values,
            title=f"📊 {i18n.t('metrics.salary_by_workclass', 'Taxa de Salário Alto por Classe Trabalhadora')}",
            template="plotly_white"
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)