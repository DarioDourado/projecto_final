"""
📁 Página de Relatórios
Sistema completo de geração e visualização de relatórios
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime
import base64

def show_reports_page(data, i18n):
    """Página principal de relatórios"""
    from src.components.navigation import show_page_header
    
    show_page_header(
        i18n.t('navigation.reports', 'Relatórios'),
        i18n.t('reports.subtitle', 'Geração e visualização de relatórios completos'),
        "📁"
    )
    
    df = data.get('df')
    if df is None or len(df) == 0:
        st.warning(i18n.t('messages.pipeline_needed', 'Execute pipeline primeiro'))
        return
    
    # Tabs para diferentes tipos de relatórios
    tab1, tab2, tab3, tab4 = st.tabs([
        f"📊 {i18n.t('reports.executive', 'Relatório Executivo')}",
        f"📈 {i18n.t('reports.detailed', 'Análise Detalhada')}",
        f"🤖 {i18n.t('reports.models', 'Relatório de Modelos')}",
        f"📁 {i18n.t('reports.export', 'Exportar Dados')}"
    ])
    
    with tab1:
        _show_executive_report(df, data, i18n)
    
    with tab2:
        _show_detailed_analysis(df, i18n)
    
    with tab3:
        _show_models_report(data, i18n)
    
    with tab4:
        _show_export_section(df, data, i18n)

def _show_executive_report(df, data, i18n):
    """Relatório executivo com KPIs principais"""
    st.subheader(f"📊 {i18n.t('reports.executive_title', 'Relatório Executivo')}")
    
    # Data do relatório
    st.markdown(f"**📅 Data do Relatório:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # KPIs Principais
    st.markdown(f"### 🎯 {i18n.t('reports.key_metrics', 'Métricas Principais')}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(df)
        st.metric(
            f"📊 {i18n.t('reports.total_records', 'Total de Registros')}",
            f"{total_records:,}"
        )
    
    with col2:
        if 'salary' in df.columns:
            high_salary_rate = (df['salary'] == '>50K').mean() * 100
            st.metric(
                f"💰 {i18n.t('reports.high_salary_rate', 'Taxa Salário >50K')}",
                f"{high_salary_rate:.1f}%"
            )
    
    with col3:
        if 'age' in df.columns:
            avg_age = df['age'].mean()
            st.metric(
                f"🎂 {i18n.t('reports.avg_age', 'Idade Média')}",
                f"{avg_age:.1f} anos"
            )
    
    with col4:
        models = data.get('models', {})
        best_accuracy = 0
        if models:
            best_accuracy = max([m.get('accuracy', 0) for m in models.values() if isinstance(m, dict)])
        st.metric(
            f"🎯 {i18n.t('reports.best_model_accuracy', 'Melhor Modelo')}",
            f"{best_accuracy:.1%}"
        )
    
    # Gráficos principais
    col1, col2 = st.columns(2)
    
    with col1:
        if 'salary' in df.columns:
            salary_dist = df['salary'].value_counts()
            fig = px.pie(
                values=salary_dist.values,
                names=salary_dist.index,
                title=f"💰 {i18n.t('reports.salary_distribution', 'Distribuição Salarial')}",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'education' in df.columns:
            education_dist = df['education'].value_counts().head(8)
            fig = px.bar(
                x=education_dist.values,
                y=education_dist.index,
                orientation='h',
                title=f"🎓 {i18n.t('reports.education_distribution', 'Distribuição Educacional')}",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Insights principais
    _show_key_insights(df, i18n)

def _show_detailed_analysis(df, i18n):
    """Análise detalhada dos dados"""
    st.subheader(f"📈 {i18n.t('reports.detailed_title', 'Análise Detalhada')}")
    
    # Seletor de variável para análise
    analysis_variable = st.selectbox(
        f"🎯 {i18n.t('reports.select_variable', 'Selecionar Variável para Análise')}:",
        df.columns.tolist()
    )
    
    if analysis_variable:
        col1, col2 = st.columns(2)
        
        with col1:
            # Estatísticas descritivas
            st.markdown(f"### 📊 {i18n.t('reports.descriptive_stats', 'Estatísticas Descritivas')}")
            
            if df[analysis_variable].dtype in ['int64', 'float64']:
                stats = df[analysis_variable].describe()
                st.dataframe(stats.to_frame().T, use_container_width=True)
                
                # Histograma
                fig = px.histogram(
                    df, x=analysis_variable,
                    title=f"📊 Distribuição de {analysis_variable}",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                value_counts = df[analysis_variable].value_counts()
                st.dataframe(value_counts.to_frame(), use_container_width=True)
                
                # Gráfico de barras
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"📊 Contagem de {analysis_variable}",
                    template="plotly_white"
                )
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Análise cruzada com salary (se existir)
            if 'salary' in df.columns and analysis_variable != 'salary':
                st.markdown(f"### 💰 {i18n.t('reports.salary_analysis', 'Análise vs Salário')}")
                
                if df[analysis_variable].dtype in ['int64', 'float64']:
                    # Box plot para variáveis numéricas
                    fig = px.box(
                        df, x='salary', y=analysis_variable,
                        title=f"📊 {analysis_variable} por Faixa Salarial",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Stacked bar para variáveis categóricas
                    crosstab = pd.crosstab(df[analysis_variable], df['salary'], normalize='index') * 100
                    
                    fig = px.bar(
                        crosstab, 
                        title=f"📊 {analysis_variable} vs Salário (%)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Matriz de correlação (se houver variáveis numéricas)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        st.markdown(f"### 🔗 {i18n.t('reports.correlation_matrix', 'Matriz de Correlação')}")
        
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title=f"🔗 {i18n.t('reports.correlation_heatmap', 'Mapa de Correlação')}",
            template="plotly_white",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig, use_container_width=True)

def _show_models_report(data, i18n):
    """Relatório específico dos modelos"""
    st.subheader(f"🤖 {i18n.t('reports.models_title', 'Relatório de Modelos')}")
    
    models = data.get('models', {})
    
    if not models:
        st.info(i18n.t('reports.no_models', 'Nenhum modelo encontrado. Execute o pipeline primeiro.'))
        return
    
    # Resumo dos modelos
    st.markdown(f"### 📊 {i18n.t('reports.models_summary', 'Resumo dos Modelos')}")
    
    model_summary = []
    for model_name, model_data in models.items():
        if isinstance(model_data, dict):
            model_summary.append({
                'Modelo': model_name,
                'Accuracy': f"{model_data.get('accuracy', 0):.3f}",
                'Precision': f"{model_data.get('precision', 0):.3f}",
                'Recall': f"{model_data.get('recall', 0):.3f}",
                'F1-Score': f"{model_data.get('f1', 0):.3f}",
                'Tempo Treino (s)': f"{model_data.get('training_time', 0):.2f}",
                'Tipo': model_data.get('model_type', 'N/A')
            })
    
    if model_summary:
        summary_df = pd.DataFrame(model_summary)
        st.dataframe(summary_df, use_container_width=True)
        
        # Gráfico de comparação
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        comparison_data = []
        
        for model_name, model_data in models.items():
            if isinstance(model_data, dict):
                for metric in metrics:
                    comparison_data.append({
                        'Modelo': model_name,
                        'Métrica': metric,
                        'Valor': model_data.get(metric.lower(), 0)
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            fig = px.bar(
                comparison_df,
                x='Modelo',
                y='Valor',
                color='Métrica',
                barmode='group',
                title=f"📊 {i18n.t('reports.models_comparison', 'Comparação de Modelos')}",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (se disponível)
    feature_importance = data.get('feature_importance', {})
    if feature_importance:
        st.markdown(f"### 📈 {i18n.t('reports.feature_importance', 'Importância das Features')}")
        
        selected_model = st.selectbox(
            f"🤖 {i18n.t('reports.select_model', 'Selecionar Modelo')}:",
            list(feature_importance.keys())
        )
        
        if selected_model and selected_model in feature_importance:
            importance_data = feature_importance[selected_model]
            
            if isinstance(importance_data, pd.DataFrame):
                # Top 10 features mais importantes
                top_features = importance_data.nlargest(10, 'Importância')
                
                fig = px.bar(
                    top_features,
                    x='Importância',
                    y='Feature',
                    orientation='h',
                    title=f"📈 Top 10 Features - {selected_model}",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

def _show_export_section(df, data, i18n):
    """Seção de exportação de dados"""
    st.subheader(f"📁 {i18n.t('reports.export_title', 'Exportar Dados')}")
    
    # Opções de exportação
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 📊 {i18n.t('reports.data_export', 'Exportar Dados')}")
        
        # Seleção de dados para exportar
        export_option = st.radio(
            f"📋 {i18n.t('reports.export_options', 'Opções de Exportação')}:",
            [
                i18n.t('reports.full_dataset', 'Dataset Completo'),
                i18n.t('reports.filtered_data', 'Dados Filtrados'),
                i18n.t('reports.summary_stats', 'Estatísticas Resumidas')
            ]
        )
        
        if export_option == i18n.t('reports.full_dataset', 'Dataset Completo'):
            csv_data = df.to_csv(index=False)
            st.download_button(
                label=f"📥 {i18n.t('reports.download_csv', 'Download CSV')}",
                data=csv_data,
                file_name=f"dataset_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        elif export_option == i18n.t('reports.summary_stats', 'Estatísticas Resumidas'):
            # Criar relatório de estatísticas
            summary_report = _create_summary_report(df, data, i18n)
            
            st.download_button(
                label=f"📥 {i18n.t('reports.download_report', 'Download Relatório')}",
                data=summary_report,
                file_name=f"relatorio_resumo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col2:
        st.markdown(f"### 📈 {i18n.t('reports.models_export', 'Exportar Modelos')}")
        
        models = data.get('models', {})
        if models:
            # Seleção de modelo para exportar
            selected_model = st.selectbox(
                f"🤖 {i18n.t('reports.select_model_export', 'Selecionar Modelo')}:",
                list(models.keys())
            )
            
            if selected_model:
                model_info = models[selected_model]
                
                # Criar JSON com informações do modelo
                model_export = {
                    'model_name': selected_model,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': {
                        'accuracy': model_info.get('accuracy', 0),
                        'precision': model_info.get('precision', 0),
                        'recall': model_info.get('recall', 0),
                        'f1': model_info.get('f1', 0),
                        'training_time': model_info.get('training_time', 0)
                    },
                    'model_type': model_info.get('model_type', 'Unknown')
                }
                
                model_json = json.dumps(model_export, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label=f"📥 {i18n.t('reports.download_model_info', 'Download Info Modelo')}",
                    data=model_json,
                    file_name=f"modelo_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info(i18n.t('reports.no_models_export', 'Nenhum modelo disponível para exportação'))

def _show_key_insights(df, i18n):
    """Mostrar insights principais dos dados"""
    st.markdown(f"### 💡 {i18n.t('reports.key_insights', 'Insights Principais')}")
    
    insights = []
    
    # Análise de salário por gênero
    if 'sex' in df.columns and 'salary' in df.columns:
        male_high_salary = (df[df['sex'] == 'Male']['salary'] == '>50K').mean()
        female_high_salary = (df[df['sex'] == 'Female']['salary'] == '>50K').mean()
        gender_gap = (male_high_salary - female_high_salary) * 100
        
        if gender_gap > 10:
            insights.append(f"⚠️ Existe um gap salarial significativo de {gender_gap:.1f} pontos percentuais entre homens e mulheres")
    
    # Análise de educação
    if 'education' in df.columns and 'salary' in df.columns:
        education_impact = df.groupby('education')['salary'].apply(lambda x: (x == '>50K').mean())
        best_education = education_impact.idxmax()
        best_rate = education_impact.max() * 100
        
        insights.append(f"🎓 {best_education} tem a maior taxa de salários altos ({best_rate:.1f}%)")
    
    # Análise de idade
    if 'age' in df.columns and 'salary' in df.columns:
        avg_age_high = df[df['salary'] == '>50K']['age'].mean()
        avg_age_low = df[df['salary'] == '<=50K']['age'].mean()
        age_diff = avg_age_high - avg_age_low
        
        if age_diff > 5:
            insights.append(f"👥 Pessoas com salários altos são em média {age_diff:.1f} anos mais velhas")
    
    # Mostrar insights
    for insight in insights:
        st.markdown(f"- {insight}")
    
    if not insights:
        st.info(i18n.t('reports.no_insights', 'Execute análises mais detalhadas para gerar insights'))

def _create_summary_report(df, data, i18n):
    """Criar relatório resumido em texto"""
    report_lines = [
        f"=== {i18n.t('reports.summary_report_title', 'RELATÓRIO RESUMO DE ANÁLISE SALARIAL')} ===",
        f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        "",
        f"=== {i18n.t('reports.dataset_info', 'INFORMAÇÕES DO DATASET')} ===",
        f"Total de registros: {len(df):,}",
        f"Total de colunas: {len(df.columns)}",
        f"Memória utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
        "",
        f"=== {i18n.t('reports.data_quality', 'QUALIDADE DOS DADOS')} ===",
        f"Completude: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%",
        f"Duplicatas: {df.duplicated().sum():,} ({df.duplicated().mean()*100:.1f}%)",
        "",
    ]
    
    # Adicionar informações de modelos se disponíveis
    models = data.get('models', {})
    if models:
        report_lines.extend([
            f"=== {i18n.t('reports.models_performance', 'PERFORMANCE DOS MODELOS')} ===",
        ])
        
        for model_name, model_data in models.items():
            if isinstance(model_data, dict):
                report_lines.append(f"{model_name}:")
                report_lines.append(f"  - Accuracy: {model_data.get('accuracy', 0):.3f}")
                report_lines.append(f"  - F1-Score: {model_data.get('f1', 0):.3f}")
                report_lines.append(f"  - Tempo: {model_data.get('training_time', 0):.2f}s")
    
    return "\n".join(report_lines)