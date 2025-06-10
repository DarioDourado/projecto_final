"""
🤖 Página de Modelos ML
Interface para visualização e análise dos modelos treinados
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

def show_models_page(data, i18n):
    """Página principal de modelos ML"""
    # Import safe
    try:
        from src.components.navigation import show_page_header
    except ImportError:
        def show_page_header(title, subtitle, icon):
            st.markdown(f"## {icon} {title}")
            if subtitle:
                st.markdown(f"*{subtitle}*")
    
    show_page_header(
        i18n.t('navigation.models', 'Modelos ML'),
        i18n.t('models.subtitle', 'Análise e comparação dos modelos de Machine Learning'),
        "🤖"
    )
    
    df = data.get('df')
    models = data.get('models', {})
    
    if df is None:
        st.warning(i18n.t('messages.pipeline_needed', 'Execute pipeline primeiro'))
        return
    
    # Tabs para diferentes aspectos dos modelos
    tab1, tab2, tab3, tab4 = st.tabs([
        f"📊 {i18n.t('models.performance', 'Performance')}",
        f"🎯 {i18n.t('models.feature_importance', 'Importância Features')}",
        f"📈 {i18n.t('models.comparison', 'Comparação')}",
        f"🔍 {i18n.t('models.details', 'Detalhes')}"
    ])
    
    with tab1:
        _show_model_performance(models, i18n)
    
    with tab2:
        _show_feature_importance(data, i18n)
    
    with tab3:
        _show_model_comparison(models, i18n)
    
    with tab4:
        _show_model_details(models, data, i18n)

def _show_model_performance(models, i18n):
    """Mostrar performance dos modelos"""
    st.subheader(f"📊 {i18n.t('models.performance_title', 'Performance dos Modelos')}")
    
    if not models:
        st.info(i18n.t('models.no_models', 'Nenhum modelo encontrado. Execute o pipeline primeiro.'))
        return
    
    # Extrair métricas dos modelos
    performance_data = []
    for model_name, model_info in models.items():
        if isinstance(model_info, dict):
            metrics = {
                'Modelo': model_name,
                'Accuracy': model_info.get('accuracy', 0),
                'Precision': model_info.get('precision', 0),
                'Recall': model_info.get('recall', 0),
                'F1-Score': model_info.get('f1', model_info.get('f1_score', 0))
            }
            performance_data.append(metrics)
    
    if performance_data:
        df_performance = pd.DataFrame(performance_data)
        
        # Mostrar tabela
        st.dataframe(df_performance.round(4), use_container_width=True)
        
        # Gráfico de barras para comparação
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig = go.Figure()
        
        for metric in metrics_to_plot:
            if metric in df_performance.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=df_performance['Modelo'],
                    y=df_performance[metric],
                    text=df_performance[metric].round(3),
                    textposition='auto'
                ))
        
        fig.update_layout(
            title="Comparação de Métricas entre Modelos",
            xaxis_title="Modelos",
            yaxis_title="Score",
            barmode='group',
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Destacar melhor modelo
        if len(df_performance) > 1:
            best_model_idx = df_performance['Accuracy'].idxmax()
            best_model = df_performance.loc[best_model_idx]
            
            st.success(f"🏆 **Melhor Modelo**: {best_model['Modelo']} com {best_model['Accuracy']:.4f} de accuracy")

def _show_feature_importance(data, i18n):
    """Mostrar importância das features"""
    st.subheader(f"🎯 {i18n.t('models.feature_importance_title', 'Importância das Features')}")
    
    # Procurar arquivos de importância de features
    importance_files = []
    
    # Verificar diferentes locais possíveis
    search_paths = [
        Path("output/analysis"),
        Path("analysis"),
        Path(".")
    ]
    
    for path in search_paths:
        if path.exists():
            importance_files.extend(list(path.glob("*feature_importance*")))
            importance_files.extend(list(path.glob("*importance*")))
    
    if importance_files:
        for file in importance_files:
            try:
                if file.suffix == '.csv':
                    importance_df = pd.read_csv(file)
                    
                    st.markdown(f"#### 📊 {file.name}")
                    
                    # Verificar se tem as colunas necessárias
                    if 'feature' in importance_df.columns.str.lower():
                        feature_col = [col for col in importance_df.columns if 'feature' in col.lower()][0]
                        importance_col = [col for col in importance_df.columns if 'importance' in col.lower() or 'score' in col.lower()]
                        
                        if importance_col:
                            importance_col = importance_col[0]
                            
                            # Ordenar por importância
                            importance_df = importance_df.sort_values(importance_col, ascending=False).head(15)
                            
                            # Criar gráfico
                            fig = px.bar(
                                importance_df,
                                x=importance_col,
                                y=feature_col,
                                orientation='h',
                                title=f"Top 15 Features - {file.stem}",
                                template="plotly_white"
                            )
                            
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Mostrar tabela
                            st.dataframe(importance_df, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Erro ao carregar {file}: {e}")
    
    else:
        st.info("Nenhum arquivo de importância de features encontrado.")
        
        # Tentar carregar modelo diretamente se disponível
        model_files = list(Path(".").glob("*random_forest*.joblib"))
        if model_files:
            try:
                model = joblib.load(model_files[0])
                if hasattr(model, 'feature_importances_'):
                    _show_direct_feature_importance(model, i18n)
            except Exception as e:
                st.error(f"Erro ao carregar modelo: {e}")

def _show_direct_feature_importance(model, i18n):
    """Mostrar importância direta do modelo"""
    st.markdown("#### 🤖 Importância Direct do Modelo")
    
    try:
        importances = model.feature_importances_
        
        # Tentar obter nomes das features
        try:
            feature_info = joblib.load("feature_info.joblib")
            feature_names = feature_info.get('feature_names', [f'Feature_{i}' for i in range(len(importances))])
        except:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Criar DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15)
        
        # Criar gráfico
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 15 Features - Random Forest",
            template="plotly_white"
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar tabela
        st.dataframe(importance_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erro ao processar importância das features: {e}")

def _show_model_comparison(models, i18n):
    """Comparação detalhada entre modelos"""
    st.subheader(f"📈 {i18n.t('models.comparison_title', 'Comparação Detalhada')}")
    
    if len(models) < 2:
        st.info("Pelo menos 2 modelos são necessários para comparação.")
        return
    
    # Criar gráfico radar para comparação
    try:
        model_data = []
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for model_name, model_info in models.items():
            if isinstance(model_info, dict):
                model_metrics = []
                for metric in metrics:
                    value = model_info.get(metric, model_info.get(metric.replace('_score', ''), 0))
                    model_metrics.append(value)
                
                model_data.append({
                    'model': model_name,
                    'metrics': model_metrics
                })
        
        if model_data:
            fig = go.Figure()
            
            for model in model_data:
                # Fechar o polígono repetindo o primeiro valor
                values = model['metrics'] + [model['metrics'][0]]
                theta = metrics + [metrics[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=theta,
                    fill='toself',
                    name=model['model'],
                    opacity=0.7
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title="Comparação Radar dos Modelos",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Erro na comparação: {e}")

def _show_model_details(models, data, i18n):
    """Detalhes específicos dos modelos"""
    st.subheader(f"🔍 {i18n.t('models.details_title', 'Detalhes dos Modelos')}")
    
    if not models:
        st.info("Nenhum modelo disponível.")
        return
    
    # Seletor de modelo
    model_names = list(models.keys())
    selected_model = st.selectbox("Selecionar Modelo:", model_names)
    
    if selected_model and selected_model in models:
        model_info = models[selected_model]
        
        if isinstance(model_info, dict):
            # Mostrar métricas em cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                accuracy = model_info.get('accuracy', 0)
                st.metric("🎯 Accuracy", f"{accuracy:.4f}")
            
            with col2:
                precision = model_info.get('precision', 0)
                st.metric("🎯 Precision", f"{precision:.4f}")
            
            with col3:
                recall = model_info.get('recall', 0)
                st.metric("📊 Recall", f"{recall:.4f}")
            
            with col4:
                f1 = model_info.get('f1', model_info.get('f1_score', 0))
                st.metric("⚖️ F1-Score", f"{f1:.4f}")
            
            # Informações adicionais se disponíveis
            st.markdown("#### ℹ️ Informações Adicionais")
            
            additional_info = {}
            for key, value in model_info.items():
                if key not in ['accuracy', 'precision', 'recall', 'f1', 'f1_score']:
                    additional_info[key] = value
            
            if additional_info:
                for key, value in additional_info.items():
                    if isinstance(value, (int, float)):
                        st.write(f"**{key.replace('_', ' ').title()}**: {value:.4f}")
                    else:
                        st.write(f"**{key.replace('_', ' ').title()}**: {value}")
            
            # Mostrar arquivos relacionados
            _show_related_files(selected_model, i18n)

def _show_related_files(model_name, i18n):
    """Mostrar arquivos relacionados ao modelo"""
    st.markdown("#### 📁 Arquivos Relacionados")
    
    # Procurar arquivos do modelo
    model_files = []
    
    # Padrões de busca
    patterns = [
        f"*{model_name.lower().replace(' ', '_')}*",
        f"*{model_name.lower()}*",
        "*.joblib",
        "*.pkl"
    ]
    
    for pattern in patterns:
        model_files.extend(list(Path(".").glob(pattern)))
    
    # Remover duplicatas
    model_files = list(set(model_files))
    
    if model_files:
        for file in model_files:
            if file.exists():
                size_mb = file.stat().st_size / (1024 * 1024)
                st.write(f"📄 **{file.name}** ({size_mb:.2f} MB)")
    else:
        st.info("Nenhum arquivo específico encontrado para este modelo.")