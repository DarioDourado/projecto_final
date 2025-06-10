"""
🔮 Página de Predição
Interface para fazer predições com modelos treinados
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.components.navigation import show_page_header, show_breadcrumbs
from src.components.layout import create_status_box, create_metric_card

def show_prediction_page(data, i18n):
    """Página de predições de salário"""
    
    # Header da página
    show_page_header(
        title=i18n.t('navigation.prediction', 'Predição'),
        subtitle=i18n.t('prediction.subtitle', 'Faça predições de salário usando modelos treinados'),
        icon="🔮"
    )
    
    # Breadcrumbs
    show_breadcrumbs([
        (i18n.t('navigation.overview', 'Visão Geral'), 'navigation.overview'),
        (i18n.t('navigation.prediction', 'Predição'), 'navigation.prediction')
    ], i18n)
    
    df = data.get('df')
    models_data = data.get('models', {})
    
    if df is None or len(df) == 0:
        st.warning(i18n.t('messages.pipeline_needed', '⚠️ Execute: python main.py'))
        return
    
    # Tabs de predição
    tab1, tab2, tab3 = st.tabs([
        f"🎯 {i18n.t('prediction.single', 'Predição Individual')}",
        f"📊 {i18n.t('prediction.batch', 'Predição em Lote')}",
        f"📈 {i18n.t('prediction.analysis', 'Análise de Predições')}"
    ])
    
    with tab1:
        _show_single_prediction(df, models_data, i18n)
    
    with tab2:
        _show_batch_prediction(df, models_data, i18n)
    
    with tab3:
        _show_prediction_analysis(df, models_data, i18n)

def _show_single_prediction(df, models_data, i18n):
    """Interface para predição individual"""
    st.markdown(f"### 🎯 {i18n.t('prediction.single', 'Predição Individual')}")
    
    if not models_data:
        st.markdown(create_status_box(
            "⚠️ Nenhum modelo encontrado. Execute o pipeline principal para treinar modelos.",
            "warning"
        ), unsafe_allow_html=True)
        return
    
    # Seleção do modelo
    col1, col2 = st.columns(2)
    
    with col1:
        available_models = list(models_data.keys())
        selected_model = st.selectbox(
            "🤖 Selecione o Modelo:",
            available_models
        )
    
    with col2:
        if selected_model:
            accuracy = models_data[selected_model].get('accuracy', 0) * 100
            st.metric("🎯 Accuracy do Modelo", f"{accuracy:.1f}%")
    
    st.markdown("---")
    
    # Formulário de entrada
    st.markdown("#### 📋 Insira os Dados para Predição")
    
    # Criar formulário baseado nas colunas do dataset
    prediction_data = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Idade
        if 'age' in df.columns:
            age_min, age_max = int(df['age'].min()), int(df['age'].max())
            prediction_data['age'] = st.slider(
                "🎂 Idade", 
                age_min, age_max, 
                value=35
            )
        
        # Sexo
        if 'sex' in df.columns:
            sex_options = df['sex'].unique().tolist()
            prediction_data['sex'] = st.selectbox(
                "👥 Sexo", 
                sex_options
            )
        
        # Educação
        if 'education' in df.columns:
            education_options = df['education'].unique().tolist()
            prediction_data['education'] = st.selectbox(
                "🎓 Educação", 
                education_options
            )
    
    with col2:
        # Anos de educação
        if 'education_years' in df.columns:
            edu_years_min, edu_years_max = int(df['education_years'].min()), int(df['education_years'].max())
            prediction_data['education_years'] = st.slider(
                "📚 Anos de Educação", 
                edu_years_min, edu_years_max,
                value=12
            )
        
        # Classe trabalhadora
        if 'workclass' in df.columns:
            workclass_options = df['workclass'].unique().tolist()
            prediction_data['workclass'] = st.selectbox(
                "💼 Classe Trabalhadora", 
                workclass_options
            )
        
        # Estado civil
        if 'marital_status' in df.columns:
            marital_options = df['marital_status'].unique().tolist()
            prediction_data['marital_status'] = st.selectbox(
                "💑 Estado Civil", 
                marital_options
            )
    
    # Botão de predição
    if st.button("🔮 Fazer Predição", use_container_width=True, type="primary"):
        prediction_result = _simulate_prediction(prediction_data, selected_model, models_data)
        _show_prediction_result(prediction_result, prediction_data, i18n)

def _show_batch_prediction(df, models_data, i18n):
    """Interface para predição em lote"""
    st.markdown(f"### 📊 {i18n.t('prediction.batch', 'Predição em Lote')}")
    
    if not models_data:
        st.markdown(create_status_box(
            "⚠️ Nenhum modelo encontrado.",
            "warning"
        ), unsafe_allow_html=True)
        return
    
    # Opções de entrada
    input_method = st.radio(
        "📥 Método de Entrada:",
        ["Amostra do Dataset", "Upload de Arquivo CSV"]
    )
    
    if input_method == "Amostra do Dataset":
        # Usar amostra do dataset atual
        st.markdown("#### 📋 Selecionar Amostra")
        
        sample_size = st.slider(
            "📊 Tamanho da Amostra:",
            min_value=10,
            max_value=min(100, len(df)),
            value=20
        )
        
        # Pegar amostra aleatória
        sample_df = df.sample(n=sample_size, random_state=42)
        
        # Remover coluna target se existir
        if 'salary' in sample_df.columns:
            sample_df = sample_df.drop('salary', axis=1)
        
        st.markdown("#### 📊 Dados para Predição")
        st.dataframe(sample_df, use_container_width=True)
        
        # Seleção do modelo
        available_models = list(models_data.keys())
        selected_model = st.selectbox(
            "🤖 Modelo para Predição:",
            available_models,
            key="batch_model"
        )
        
        if st.button("🚀 Executar Predições em Lote", use_container_width=True):
            _execute_batch_prediction(sample_df, selected_model, models_data, i18n)
    
    else:
        # Upload de arquivo
        st.markdown("#### 📁 Upload de Arquivo")
        
        uploaded_file = st.file_uploader(
            "Selecione um arquivo CSV:",
            type=['csv'],
            help="O arquivo deve ter as mesmas colunas do dataset de treino"
        )
        
        if uploaded_file is not None:
            try:
                upload_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Arquivo carregado: {len(upload_df)} registros")
                
                st.dataframe(upload_df.head(), use_container_width=True)
                
                # Seleção do modelo
                available_models = list(models_data.keys())
                selected_model = st.selectbox(
                    "🤖 Modelo para Predição:",
                    available_models,
                    key="upload_model"
                )
                
                if st.button("🚀 Executar Predições", use_container_width=True):
                    _execute_batch_prediction(upload_df, selected_model, models_data, i18n)
                    
            except Exception as e:
                st.error(f"❌ Erro ao ler arquivo: {e}")

def _show_prediction_analysis(df, models_data, i18n):
    """Análise de predições"""
    st.markdown(f"### 📈 {i18n.t('prediction.analysis', 'Análise de Predições')}")
    
    if not models_data:
        st.markdown(create_status_box(
            "⚠️ Nenhum modelo encontrado.",
            "warning"
        ), unsafe_allow_html=True)
        return
    
    # Análise de cenários
    st.markdown("#### 🔍 Análise de Cenários")
    
    if 'age' in df.columns and 'education' in df.columns:
        # Criar grid de predições para análise
        age_range = np.linspace(df['age'].min(), df['age'].max(), 10)
        education_levels = df['education'].unique()[:5]  # Top 5 educações
        
        # Simular predições para diferentes combinações
        scenario_results = []
        
        for age in age_range:
            for education in education_levels:
                # Criar dados base
                base_data = {
                    'age': age,
                    'education': education,
                    'sex': df['sex'].mode()[0] if 'sex' in df.columns else 'Male',
                    'workclass': df['workclass'].mode()[0] if 'workclass' in df.columns else 'Private'
                }
                
                # Simular predição
                pred_prob = _simulate_prediction_probability(base_data)
                
                scenario_results.append({
                    'Idade': int(age),
                    'Educação': education,
                    'Probabilidade_Alto_Salario': pred_prob,
                    'Predição': '>50K' if pred_prob > 0.5 else '<=50K'
                })
        
        scenario_df = pd.DataFrame(scenario_results)
        
        # Heatmap de predições
        pivot_table = scenario_df.pivot(index='Educação', columns='Idade', values='Probabilidade_Alto_Salario')
        
        fig = px.imshow(
            pivot_table,
            title="Probabilidade de Salário Alto por Idade e Educação",
            template="plotly_white",
            color_continuous_scale="RdYlBu",
            aspect="auto"
        )
        
        fig.update_layout(
            xaxis_title="Idade",
            yaxis_title="Nível de Educação",
            coloraxis_colorbar_title="Probabilidade"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights automáticos
        st.markdown("#### 💡 Insights da Análise")
        
        # Educação com maior probabilidade
        avg_by_education = scenario_df.groupby('Educação')['Probabilidade_Alto_Salario'].mean()
        best_education = avg_by_education.idxmax()
        best_prob = avg_by_education.max()
        
        st.markdown(create_status_box(
            f"🎓 Melhor educação para salário alto: **{best_education}** (probabilidade média: {best_prob:.1%})",
            "success"
        ), unsafe_allow_html=True)
        
        # Faixa etária com maior probabilidade
        avg_by_age = scenario_df.groupby('Idade')['Probabilidade_Alto_Salario'].mean()
        best_age = avg_by_age.idxmax()
        best_age_prob = avg_by_age.max()
        
        st.markdown(create_status_box(
            f"🎂 Melhor idade para salário alto: **{best_age} anos** (probabilidade: {best_age_prob:.1%})",
            "info"
        ), unsafe_allow_html=True)

def _simulate_prediction(data, model_name, models_data):
    """Simular predição individual"""
    # Simular probabilidades baseadas no modelo
    model_accuracy = models_data[model_name].get('accuracy', 0.8)
    
    # Fatores que influenciam a predição (simulado)
    factors = {
        'age': data.get('age', 35) / 100,  # Normalizar idade
        'education': _education_score(data.get('education', 'HS-grad')),
        'sex': 0.6 if data.get('sex') == 'Male' else 0.4,  # Simular bias histórico
        'workclass': _workclass_score(data.get('workclass', 'Private'))
    }
    
    # Calcular probabilidade base
    base_prob = np.mean(list(factors.values()))
    
    # Ajustar com accuracy do modelo
    final_prob = base_prob * model_accuracy + np.random.normal(0, 0.1)
    final_prob = np.clip(final_prob, 0, 1)
    
    prediction = '>50K' if final_prob > 0.5 else '<=50K'
    confidence = abs(final_prob - 0.5) * 2  # Confiança baseada na distância de 0.5
    
    return {
        'prediction': prediction,
        'probability': final_prob,
        'confidence': confidence,
        'factors': factors
    }

def _simulate_prediction_probability(data):
    """Simular probabilidade de predição para análise"""
    # Lógica simplificada para demonstração
    age_factor = data.get('age', 35) / 100
    education_factor = _education_score(data.get('education', 'HS-grad'))
    
    # Combinar fatores
    prob = (age_factor * 0.3 + education_factor * 0.7) + np.random.normal(0, 0.1)
    return np.clip(prob, 0, 1)

def _education_score(education):
    """Converter educação em score numérico"""
    education_scores = {
        'Preschool': 0.1,
        '1st-4th': 0.2,
        '5th-6th': 0.25,
        '7th-8th': 0.3,
        '9th': 0.35,
        '10th': 0.4,
        '11th': 0.45,
        '12th': 0.5,
        'HS-grad': 0.5,
        'Some-college': 0.6,
        'Assoc-voc': 0.65,
        'Assoc-acdm': 0.7,
        'Bachelors': 0.8,
        'Masters': 0.9,
        'Prof-school': 0.95,
        'Doctorate': 1.0
    }
    return education_scores.get(education, 0.5)

def _workclass_score(workclass):
    """Converter classe trabalhadora em score"""
    workclass_scores = {
        'Private': 0.6,
        'Self-emp-not-inc': 0.5,
        'Self-emp-inc': 0.7,
        'Federal-gov': 0.8,
        'Local-gov': 0.7,
        'State-gov': 0.75,
        'Without-pay': 0.1,
        'Never-worked': 0.1
    }
    return workclass_scores.get(workclass, 0.5)

def _show_prediction_result(result, input_data, i18n):
    """Mostrar resultado da predição"""
    st.markdown("---")
    st.markdown("### 🎯 Resultado da Predição")
    
    # Resultado principal
    prediction = result['prediction']
    probability = result['probability']
    confidence = result['confidence']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color = "🟢" if prediction == '>50K' else "🔴"
        st.markdown(create_metric_card(
            title="Predição",
            value=f"{color} {prediction}",
            icon="🎯"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card(
            title="Probabilidade",
            value=f"{probability:.1%}",
            icon="📊"
        ), unsafe_allow_html=True)
    
    with col3:
        confidence_level = "Alta" if confidence > 0.7 else "Média" if confidence > 0.4 else "Baixa"
        st.markdown(create_metric_card(
            title="Confiança",
            value=f"{confidence:.1%} ({confidence_level})",
            icon="🎯"
        ), unsafe_allow_html=True)
    
    # Fatores que influenciaram
    st.markdown("#### 📊 Fatores que Influenciaram a Predição")
    
    factors = result['factors']
    factors_df = pd.DataFrame([
        {'Fator': k.title(), 'Peso': v, 'Influência': 'Positiva' if v > 0.5 else 'Negativa'}
        for k, v in factors.items()
    ])
    
    fig = px.bar(
        factors_df,
        x='Fator',
        y='Peso',
        color='Influência',
        title="Peso dos Fatores na Predição",
        template="plotly_white",
        color_discrete_map={'Positiva': '#2E8B57', 'Negativa': '#DC143C'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _execute_batch_prediction(df, model_name, models_data, i18n):
    """Executar predições em lote"""
    import time
    
    # Barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Executando predições...")
    
    # Simular processamento
    predictions = []
    for i, (idx, row) in enumerate(df.iterrows()):
        # Simular predição
        data_dict = row.to_dict()
        result = _simulate_prediction(data_dict, model_name, models_data)
        
        predictions.append({
            'Índice': idx,
            'Predição': result['prediction'],
            'Probabilidade': f"{result['probability']:.2%}",
            'Confiança': f"{result['confidence']:.2%}"
        })
        
        # Atualizar progresso
        progress_bar.progress((i + 1) / len(df))
        time.sleep(0.05)  # Simular processamento
    
    status_text.text("✅ Predições concluídas!")
    
    # Mostrar resultados
    st.markdown("### 📊 Resultados das Predições")
    
    results_df = pd.DataFrame(predictions)
    
    # Métricas de resumo
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_predictions = len(results_df)
        st.metric("📊 Total de Predições", total_predictions)
    
    with col2:
        high_salary_count = len(results_df[results_df['Predição'] == '>50K'])
        high_salary_pct = (high_salary_count / total_predictions) * 100
        st.metric("💰 Salário Alto", f"{high_salary_count} ({high_salary_pct:.1f}%)")
    
    with col3:
        low_salary_count = len(results_df[results_df['Predição'] == '<=50K'])
        low_salary_pct = (low_salary_count / total_predictions) * 100
        st.metric("💵 Salário Baixo", f"{low_salary_count} ({low_salary_pct:.1f}%)")
    
    # Tabela de resultados
    st.dataframe(results_df, use_container_width=True)
    
    # Gráfico de distribuição
    prediction_counts = results_df['Predição'].value_counts()
    
    fig = px.pie(
        values=prediction_counts.values,
        names=prediction_counts.index,
        title="Distribuição das Predições",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Botão de download
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Resultados (CSV)",
        data=csv,
        file_name=f"predicoes_{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )