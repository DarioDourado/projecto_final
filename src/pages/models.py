"""
🤖 Página de Modelos de Machine Learning
Treinamento, avaliação e comparação de modelos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
from src.components.navigation import show_page_header, show_breadcrumbs
from src.components.layout import create_status_box, create_metric_card

def show_models_page(data, i18n):
    """Página de modelos de machine learning"""
    
    # Header da página
    show_page_header(
        title=i18n.t('navigation.models', 'Modelos ML'),
        subtitle=i18n.t('models.subtitle', 'Treinamento e avaliação de modelos de machine learning'),
        icon="🤖"
    )
    
    # Breadcrumbs
    show_breadcrumbs([
        (i18n.t('navigation.overview', 'Visão Geral'), 'navigation.overview'),
        (i18n.t('navigation.models', 'Modelos ML'), 'navigation.models')
    ], i18n)
    
    df = data.get('df')
    if df is None or len(df) == 0:
        st.warning(i18n.t('messages.pipeline_needed', '⚠️ Execute: python main.py'))
        return
    
    # Verificar se há modelos treinados
    models_data = data.get('models', {})
    feature_importance = data.get('feature_importance', {})
    
    # Tabs de modelos
    tab1, tab2, tab3, tab4 = st.tabs([
        f"📊 {i18n.t('models.overview', 'Visão Geral')}",
        f"🏆 {i18n.t('models.comparison', 'Comparação')}",
        f"📈 {i18n.t('models.importance', 'Importância das Features')}",
        f"🎯 {i18n.t('models.training', 'Treinar Modelos')}"
    ])
    
    with tab1:
        _show_models_overview(models_data, i18n)
    
    with tab2:
        _show_models_comparison(models_data, i18n)
    
    with tab3:
        _show_feature_importance(feature_importance, i18n)
    
    with tab4:
        _show_training_interface(df, i18n)

def _show_models_overview(models_data, i18n):
    """Visão geral dos modelos"""
    st.markdown(f"### 📊 {i18n.t('models.trained_models', 'Modelos Treinados')}")
    
    if not models_data:
        st.markdown(create_status_box(
            "⚠️ Nenhum modelo encontrado. Execute o pipeline principal para treinar modelos.",
            "warning"
        ), unsafe_allow_html=True)
        return
    
    # Métricas dos modelos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_metric_card(
            title="🤖 Modelos Treinados",
            value=str(len(models_data)),
            icon="📊"
        ), unsafe_allow_html=True)
    
    with col2:
        if models_data:
            best_model = max(models_data.values(), key=lambda x: x.get('accuracy', 0))
            best_accuracy = best_model.get('accuracy', 0) * 100
            st.markdown(create_metric_card(
                title="🏆 Melhor Accuracy",
                value=f"{best_accuracy:.1f}%",
                icon="🎯"
            ), unsafe_allow_html=True)
    
    with col3:
        if models_data:
            best_model_name = max(models_data.keys(), key=lambda x: models_data[x].get('accuracy', 0))
            st.markdown(create_metric_card(
                title="👑 Melhor Modelo",
                value=best_model_name,
                icon="🤖"
            ), unsafe_allow_html=True)
    
    # Detalhes dos modelos
    st.markdown("#### 📋 Detalhes dos Modelos")
    
    model_details = []
    for model_name, model_info in models_data.items():
        model_details.append({
            'Modelo': model_name,
            'Accuracy': f"{model_info.get('accuracy', 0)*100:.2f}%",
            'Precision': f"{model_info.get('precision', 0)*100:.2f}%",
            'Recall': f"{model_info.get('recall', 0)*100:.2f}%",
            'F1-Score': f"{model_info.get('f1', 0)*100:.2f}%",
            'Tempo de Treino': f"{model_info.get('training_time', 0):.2f}s"
        })
    
    if model_details:
        models_df = pd.DataFrame(model_details)
        st.dataframe(models_df, use_container_width=True)

def _show_models_comparison(models_data, i18n):
    """Comparação entre modelos"""
    st.markdown(f"### 🏆 {i18n.t('models.comparison', 'Comparação de Modelos')}")
    
    if not models_data:
        st.markdown(create_status_box(
            "⚠️ Nenhum modelo para comparar. Treine alguns modelos primeiro.",
            "warning"
        ), unsafe_allow_html=True)
        return
    
    # Métricas para comparação
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = {
        'accuracy': 'Accuracy',
        'precision': 'Precision', 
        'recall': 'Recall',
        'f1': 'F1-Score'
    }
    
    # Preparar dados para gráficos
    models_list = list(models_data.keys())
    
    # Gráfico de barras comparativo
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[metric_names[m] for m in metrics],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = px.colors.qualitative.Set3[:len(models_list)]
    
    for i, metric in enumerate(metrics):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        values = [models_data[model].get(metric, 0) * 100 for model in models_list]
        
        fig.add_trace(
            go.Bar(
                x=models_list,
                y=values,
                name=metric_names[metric],
                marker_color=colors,
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.update_yaxes(title_text="Percentual (%)", row=row, col=col)
        fig.update_xaxes(title_text="Modelos", row=row, col=col)
    
    fig.update_layout(
        title_text="Comparação de Métricas entre Modelos",
        height=600,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart para comparação geral
    st.markdown("#### 🕸️ Gráfico Radar - Comparação Geral")
    
    fig_radar = go.Figure()
    
    for i, model_name in enumerate(models_list):
        model_info = models_data[model_name]
        values = [model_info.get(metric, 0) * 100 for metric in metrics]
        values.append(values[0])  # Fechar o polígono
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=list(metric_names.values()) + [list(metric_names.values())[0]],
            fill='toself',
            name=model_name,
            line_color=colors[i]
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Comparação Radar dos Modelos",
        template="plotly_white"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Ranking dos modelos
    st.markdown("#### 🏅 Ranking dos Modelos")
    
    # Calcular score combinado
    model_scores = []
    for model_name, model_info in models_data.items():
        combined_score = np.mean([
            model_info.get('accuracy', 0),
            model_info.get('precision', 0),
            model_info.get('recall', 0),
            model_info.get('f1', 0)
        ]) * 100
        
        model_scores.append({
            'Posição': '',
            'Modelo': model_name,
            'Score Combinado': f"{combined_score:.2f}%",
            'Accuracy': f"{model_info.get('accuracy', 0)*100:.2f}%",
            'Melhor em': _get_best_metric(model_info)
        })
    
    # Ordenar por score combinado
    model_scores.sort(key=lambda x: float(x['Score Combinado'].replace('%', '')), reverse=True)
    
    # Adicionar posições
    for i, model in enumerate(model_scores):
        if i == 0:
            model['Posição'] = "🥇"
        elif i == 1:
            model['Posição'] = "🥈"
        elif i == 2:
            model['Posição'] = "🥉"
        else:
            model['Posição'] = f"{i+1}º"
    
    ranking_df = pd.DataFrame(model_scores)
    st.dataframe(ranking_df, use_container_width=True)

def _show_feature_importance(feature_importance, i18n):
    """Mostrar importância das features"""
    st.markdown(f"### 📈 {i18n.t('models.importance', 'Importância das Features')}")
    
    if not feature_importance:
        st.markdown(create_status_box(
            "⚠️ Dados de importância das features não encontrados.",
            "warning"
        ), unsafe_allow_html=True)
        return
    
    # Seletor de modelo
    available_models = list(feature_importance.keys())
    
    if available_models:
        selected_model = st.selectbox(
            "🤖 Selecione o Modelo:",
            available_models
        )
        
        if selected_model and selected_model in feature_importance:
            importance_data = feature_importance[selected_model]
            
            # Converter para DataFrame se necessário
            if isinstance(importance_data, dict):
                features_df = pd.DataFrame(
                    list(importance_data.items()),
                    columns=['Feature', 'Importância']
                )
            else:
                features_df = importance_data
            
            # Ordenar por importância
            features_df = features_df.sort_values('Importância', ascending=True)
            
            # Gráfico de barras horizontais
            fig = px.bar(
                features_df.tail(15),  # Top 15 features
                x='Importância',
                y='Feature',
                orientation='h',
                title=f"Importância das Features - {selected_model}",
                template="plotly_white",
                color='Importância',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                height=600,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis_title="Importância",
                yaxis_title="Features"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela de importância
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔝 Top 10 Features Mais Importantes")
                top_features = features_df.tail(10).sort_values('Importância', ascending=False)
                st.dataframe(top_features, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Estatísticas de Importância")
                st.write(f"**Total de Features:** {len(features_df)}")
                st.write(f"**Importância Média:** {features_df['Importância'].mean():.4f}")
                st.write(f"**Importância Máxima:** {features_df['Importância'].max():.4f}")
                st.write(f"**Importância Mínima:** {features_df['Importância'].min():.4f}")
                
                # Top 3 features
                st.markdown("**🏆 Top 3 Features:**")
                for i, (_, row) in enumerate(features_df.tail(3).sort_values('Importância', ascending=False).iterrows()):
                    emoji = ["🥇", "🥈", "🥉"][i]
                    st.write(f"{emoji} {row['Feature']}: {row['Importância']:.4f}")

def _show_training_interface(df, i18n):
    """Interface para treinar novos modelos"""
    st.markdown(f"### 🎯 {i18n.t('models.training', 'Treinar Novos Modelos')}")
    
    st.markdown(create_status_box(
        "ℹ️ Para treinar modelos completos, execute o pipeline principal: python main.py",
        "info"
    ), unsafe_allow_html=True)
    
    # Simulação de treinamento rápido
    st.markdown("#### ⚡ Treinamento Rápido (Demo)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_column = st.selectbox(
            "🎯 Variável Target:",
            ['salary'] if 'salary' in df.columns else df.columns.tolist()
        )
    
    with col2:
        model_type = st.selectbox(
            "🤖 Tipo de Modelo:",
            ["Random Forest", "Logistic Regression", "SVM", "Gradient Boosting"]
        )
    
    # Configurações do modelo
    st.markdown("#### ⚙️ Configurações")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("📊 Tamanho do Teste (%)", 10, 40, 20) / 100
    
    with col2:
        random_state = st.number_input("🎲 Random State", 0, 100, 42)
    
    with col3:
        cross_validation = st.checkbox("🔄 Cross Validation", value=True)
    
    # Botão de treinamento
    if st.button("🚀 Treinar Modelo", use_container_width=True):
        _simulate_model_training(df, target_column, model_type, test_size, random_state, cross_validation, i18n)

def _simulate_model_training(df, target, model_type, test_size, random_state, cv, i18n):
    """Simular treinamento de modelo"""
    import time
    
    # Barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    steps = [
        "Preparando dados...",
        "Dividindo dataset...", 
        "Treinando modelo...",
        "Avaliando performance...",
        "Finalizando..."
    ]
    
    for i, step in enumerate(steps):
        status_text.text(step)
        time.sleep(0.5)
        progress_bar.progress((i + 1) / len(steps))
    
    # Simular resultados (valores aleatórios realistas)
    np.random.seed(random_state)
    
    results = {
        'accuracy': np.random.uniform(0.75, 0.95),
        'precision': np.random.uniform(0.70, 0.90),
        'recall': np.random.uniform(0.70, 0.90),
        'f1': np.random.uniform(0.70, 0.90),
        'training_time': np.random.uniform(0.5, 3.0)
    }
    
    status_text.text("✅ Treinamento concluído!")
    
    # Mostrar resultados
    st.markdown("#### 📊 Resultados do Treinamento")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Accuracy", f"{results['accuracy']*100:.1f}%")
    
    with col2:
        st.metric("🔍 Precision", f"{results['precision']*100:.1f}%")
    
    with col3:
        st.metric("📈 Recall", f"{results['recall']*100:.1f}%")
    
    with col4:
        st.metric("⚖️ F1-Score", f"{results['f1']*100:.1f}%")
    
    st.success(f"✅ Modelo {model_type} treinado com sucesso em {results['training_time']:.2f} segundos!")

def _get_best_metric(model_info):
    """Determinar em qual métrica o modelo é melhor"""
    metrics = {
        'Accuracy': model_info.get('accuracy', 0),
        'Precision': model_info.get('precision', 0),
        'Recall': model_info.get('recall', 0),
        'F1-Score': model_info.get('f1', 0)
    }
    
    best_metric = max(metrics.keys(), key=lambda k: metrics[k])
    return best_metric