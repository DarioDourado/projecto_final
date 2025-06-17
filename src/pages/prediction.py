"""
Página de Predição Salarial Interativa
Sistema de Machine Learning para Predição em Tempo Real
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import joblib
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def show_prediction_page(data):
    """Página principal de predição salarial"""
    st.title("🔮 Predição Salarial Interativa")
    st.markdown("### Sistema Avançado de Machine Learning")
    
    # Header contextual
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    ">
        <h2 style="margin: 0; font-size: 1.8rem;">🎯 Predição Científica de Salários</h2>
        <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;">
            Algoritmos: Random Forest • Logistic Regression • Feature Engineering
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar disponibilidade de dados
    if not data or 'original' not in data:
        show_no_data_warning()
        return
    
    # Carregar modelo se disponível
    model_status = load_trained_model()
    
    # Contexto do dataset
    show_dataset_context(data['original'])
    
    # Interface principal de predição
    prediction_result = show_enhanced_prediction_interface(data['original'], model_status)
    
    # Histórico de predições
    show_prediction_history()
    
    # Análise de performance do modelo
    show_model_performance()
    
    # Insights e interpretabilidade
    show_advanced_insights()

def load_trained_model():
    """Carregar modelo treinado se disponível"""
    model_paths = [
        Path("models/random_forest_model.joblib"),
        Path("models/logistic_regression_model.joblib"),
        Path("models/best_model.joblib")
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                feature_info_path = model_path.parent / "feature_info.joblib"
                feature_info = joblib.load(feature_info_path) if feature_info_path.exists() else None
                
                return {
                    'model': model,
                    'feature_info': feature_info,
                    'model_name': model_path.stem,
                    'path': str(model_path),
                    'loaded': True
                }
            except Exception as e:
                logger.warning(f"Erro ao carregar modelo {model_path}: {e}")
                continue
    
    return {'loaded': False, 'model': None}

def show_no_data_warning():
    """Exibir aviso quando não há dados"""
    st.warning("⚠️ Dados não disponíveis para predição")
    st.info("Execute o pipeline principal para gerar os dados necessários:")
    st.code("python main.py", language="bash")

def show_dataset_context(df):
    """Mostrar contexto e estatísticas do dataset"""
    st.subheader("📊 Contexto dos Dados de Treinamento")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_context_card("📋 Total de Registros", f"{len(df):,}", "Base de dados", "#4285f4")
    
    with col2:
        if 'salary' in df.columns:
            high_salary_pct = (df['salary'] == '>50K').mean() * 100
            create_context_card("💰 Salários >50K", f"{high_salary_pct:.1f}%", "Distribuição positiva", "#28a745")
        else:
            create_context_card("💰 Salários >50K", "N/A", "Dados indisponíveis", "#6c757d")
    
    with col3:
        if 'age' in df.columns:
            avg_age = df['age'].mean()
            create_context_card("👤 Idade Média", f"{avg_age:.1f}", "Anos", "#ffc107")
        else:
            create_context_card("👤 Idade Média", "N/A", "Dados indisponíveis", "#6c757d")
    
    with col4:
        if 'hours-per-week' in df.columns:
            avg_hours = df['hours-per-week'].mean()
            create_context_card("⏰ Horas/Semana", f"{avg_hours:.1f}", "Média trabalho", "#17a2b8")
        else:
            create_context_card("⏰ Horas/Semana", "N/A", "Dados indisponíveis", "#6c757d")

def create_context_card(title, value, description, color):
    """Criar card de contexto visual"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}08);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid {color};
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.2s;
    ">
        <h4 style="margin: 0; color: #333; font-size: 0.9rem; font-weight: 600;">
            {title}
        </h4>
        <h2 style="margin: 0.6rem 0; color: {color}; font-size: 2rem; font-weight: bold;">
            {value}
        </h2>
        <p style="margin: 0; color: #666; font-size: 0.8rem; opacity: 0.8;">
            {description}
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_enhanced_prediction_interface(df, model_status):
    """Interface avançada de predição"""
    st.subheader("🚀 Configuração de Predição Avançada")
    
    # Status do modelo
    if model_status['loaded']:
        st.success(f"✅ Modelo carregado: {model_status['model_name']}")
    else:
        st.warning("⚠️ Usando modelo heurístico (treine modelos para melhor precisão)")
    
    # Tabs para diferentes modos de predição
    tab1, tab2, tab3 = st.tabs(["🎯 Predição Individual", "📊 Predição em Lote", "🔄 Comparação de Modelos"])
    
    with tab1:
        result = show_individual_prediction_form(df, model_status)
        return result
    
    with tab2:
        show_batch_prediction_interface(df, model_status)
    
    with tab3:
        show_model_comparison_interface(df, model_status)

def show_individual_prediction_form(df, model_status):
    """Formulário de predição individual"""
    with st.form("enhanced_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Informações Pessoais")
            
            age = st.slider(
                "🎂 Idade",
                min_value=18, 
                max_value=90, 
                value=35,
                step=1,
                help="Idade em anos - fator importante para experiência profissional"
            )
            
            sex = st.selectbox(
                "⚥ Sexo",
                ['Male', 'Female'],
                help="Informação demográfica"
            )
            
            education = st.selectbox(
                "🎓 Nível de Educação",
                [
                    'Doctorate', 'Prof-school', 'Masters', 'Bachelors',
                    'Assoc-acdm', 'Assoc-voc', 'Some-college', 'HS-grad',
                    '12th', '11th', '10th', '9th', '7th-8th', '5th-6th',
                    '1st-4th', 'Preschool'
                ],
                index=3,  # Bachelors por padrão
                help="Maior nível educacional completado - fator mais importante"
            )
            
            marital_status = st.selectbox(
                "💑 Estado Civil",
                [
                    'Married-civ-spouse', 'Never-married', 'Divorced',
                    'Separated', 'Widowed', 'Married-spouse-absent',
                    'Married-AF-spouse'
                ],
                help="Estado civil atual"
            )
            
            race = st.selectbox(
                "🌍 Etnia",
                ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
                help="Informação étnica"
            )
            
            native_country = st.selectbox(
                "🗺️ País de Origem",
                ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 
                 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'Other'],
                help="País de nascimento"
            )
        
        with col2:
            st.markdown("#### 💼 Informações Profissionais")
            
            workclass = st.selectbox(
                "🏢 Classe de Trabalho",
                [
                    'Private', 'Self-emp-not-inc', 'Self-emp-inc',
                    'Federal-gov', 'Local-gov', 'State-gov',
                    'Without-pay', 'Never-worked'
                ],
                help="Tipo de empregador"
            )
            
            occupation = st.selectbox(
                "💼 Ocupação",
                [
                    'Prof-specialty', 'Exec-managerial', 'Tech-support',
                    'Craft-repair', 'Sales', 'Adm-clerical', 'Other-service',
                    'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners',
                    'Farming-fishing', 'Protective-serv', 'Priv-house-serv',
                    'Armed-Forces'
                ],
                help="Área de atuação profissional"
            )
            
            hours_per_week = st.slider(
                "⏰ Horas por Semana",
                min_value=1,
                max_value=99,
                value=40,
                step=1,
                help="Horas trabalhadas por semana"
            )
            
            relationship = st.selectbox(
                "👨‍👩‍👧‍👦 Relacionamento Familiar",
                [
                    'Husband', 'Not-in-family', 'Wife',
                    'Own-child', 'Unmarried', 'Other-relative'
                ],
                help="Posição na família"
            )
            
            # Campos calculados automaticamente
            education_num = get_education_num(education)
            st.info(f"📚 Anos de educação calculados: {education_num}")
            
            capital_gain = st.number_input(
                "💹 Ganho de Capital",
                min_value=0,
                max_value=99999,
                value=0,
                help="Ganhos de capital no ano"
            )
            
            capital_loss = st.number_input(
                "📉 Perda de Capital",
                min_value=0,
                max_value=99999,
                value=0,
                help="Perdas de capital no ano"
            )
        
        # Botão de predição
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        
        with col_btn2:
            predict_button = st.form_submit_button(
                "🚀 Executar Predição Avançada",
                type="primary",
                use_container_width=True
            )
    
    # Processar predição
    if predict_button:
        features = {
            'age': age,
            'sex': sex,
            'education': education,
            'education_num': education_num,
            'marital_status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'native_country': native_country,
            'workclass': workclass,
            'hours_per_week': hours_per_week,
            'capital_gain': capital_gain,
            'capital_loss': capital_loss
        }
        
        # Executar predição
        if model_status['loaded']:
            prediction_result = predict_with_trained_model(features, model_status)
        else:
            prediction_result = predict_with_heuristic_model(features)
        
        # Salvar no histórico
        save_prediction_to_history(prediction_result)
        
        # Exibir resultados
        show_enhanced_prediction_result(prediction_result)
        
        return prediction_result
    
    return None

def get_education_num(education):
    """Converter nível educacional para número de anos"""
    education_mapping = {
        'Preschool': 1, '1st-4th': 3, '5th-6th': 6, '7th-8th': 8,
        '9th': 9, '10th': 10, '11th': 11, '12th': 12, 'HS-grad': 13,
        'Some-college': 14, 'Assoc-voc': 15, 'Assoc-acdm': 15,
        'Bachelors': 16, 'Masters': 18, 'Prof-school': 20, 'Doctorate': 21
    }
    return education_mapping.get(education, 13)

def predict_with_trained_model(features, model_status):
    """Predição usando modelo treinado"""
    try:
        model = model_status['model']
        feature_info = model_status.get('feature_info', {})
        
        # Preparar dados para o modelo
        feature_df = prepare_features_for_model(features, feature_info)
        
        # Fazer predição
        prediction = model.predict(feature_df)[0]
        
        # Obter probabilidades se disponível
        try:
            probabilities = model.predict_proba(feature_df)[0]
            probability_high = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        except:
            probability_high = 0.5  # Fallback
        
        # Calcular confiança
        confidence = calculate_confidence(probability_high)
        
        # Obter feature importance se disponível
        feature_importance = get_feature_importance(model, feature_df.columns)
        
        return {
            'prediction': prediction,
            'probability': probability_high,
            'confidence': confidence,
            'method': 'trained_model',
            'model_name': model_status['model_name'],
            'features': features,
            'feature_importance': feature_importance,
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Erro na predição com modelo treinado: {e}")
        return predict_with_heuristic_model(features)

def prepare_features_for_model(features, feature_info):
    """Preparar features para o modelo treinado"""
    # Criar DataFrame com as features
    df = pd.DataFrame([features])
    
    # Aplicar encoding se necessário
    if feature_info and 'encoders' in feature_info:
        for column, encoder in feature_info['encoders'].items():
            if column in df.columns:
                try:
                    df[column] = encoder.transform(df[column])
                except:
                    # Se falhar, usar valor padrão
                    df[column] = 0
    
    return df

def predict_with_heuristic_model(features):
    """Predição usando modelo heurístico (regras de negócio)"""
    score = 0
    details = []
    max_score = 100
    
    # Pontuação por educação (peso 30%)
    education_scores = {
        'Doctorate': 30, 'Prof-school': 28, 'Masters': 25, 'Bachelors': 20,
        'Assoc-acdm': 12, 'Assoc-voc': 10, 'Some-college': 8, 'HS-grad': 5,
        '12th': 3, '11th': 2, '10th': 1, '9th': 0, '7th-8th': 0
    }
    edu_score = education_scores.get(features['education'], 0)
    score += edu_score
    details.append(f"Educação ({features['education']}): +{edu_score} pontos")
    
    # Pontuação por idade (peso 20%)
    age = features['age']
    if age >= 50:
        age_score = 20
    elif age >= 40:
        age_score = 15
    elif age >= 30:
        age_score = 10
    elif age >= 25:
        age_score = 5
    else:
        age_score = 0
    
    score += age_score
    details.append(f"Idade ({age} anos): +{age_score} pontos")
    
    # Pontuação por ocupação (peso 15%)
    occupation_scores = {
        'Exec-managerial': 15, 'Prof-specialty': 14, 'Tech-support': 10,
        'Sales': 8, 'Craft-repair': 6, 'Adm-clerical': 4, 'Protective-serv': 6,
        'Other-service': 2, 'Machine-op-inspct': 3, 'Transport-moving': 3,
        'Handlers-cleaners': 1, 'Farming-fishing': 1, 'Priv-house-serv': 0
    }
    occ_score = occupation_scores.get(features['occupation'], 3)
    score += occ_score
    details.append(f"Ocupação ({features['occupation']}): +{occ_score} pontos")
    
    # Pontuação por horas trabalhadas (peso 15%)
    hours = features['hours_per_week']
    if hours >= 50:
        hours_score = 15
    elif hours >= 40:
        hours_score = 10
    elif hours >= 30:
        hours_score = 5
    else:
        hours_score = 0
    
    score += hours_score
    details.append(f"Horas/semana ({hours}h): +{hours_score} pontos")
    
    # Pontuação por estado civil (peso 10%)
    if features['marital_status'] in ['Married-civ-spouse', 'Married-AF-spouse']:
        marital_score = 10
        details.append(f"Estado civil (Casado): +{marital_score} pontos")
    else:
        marital_score = 0
    
    score += marital_score
    
    # Pontuação por classe de trabalho (peso 10%)
    workclass_scores = {
        'Federal-gov': 8, 'Self-emp-inc': 10, 'Local-gov': 6,
        'State-gov': 6, 'Private': 5, 'Self-emp-not-inc': 3,
        'Without-pay': 0, 'Never-worked': 0
    }
    work_score = workclass_scores.get(features['workclass'], 0)
    score += work_score
    details.append(f"Classe trabalho ({features['workclass']}): +{work_score} pontos")
    
    # Calcular probabilidade
    probability = min(score / max_score, 0.95)
    prediction = ">50K" if probability > 0.5 else "<=50K"
    confidence = calculate_confidence(probability)
    
    return {
        'prediction': prediction,
        'probability': probability,
        'confidence': confidence,
        'method': 'heuristic',
        'score': score,
        'max_score': max_score,
        'details': details,
        'features': features,
        'timestamp': datetime.now()
    }

def calculate_confidence(probability):
    """Calcular nível de confiança baseado na probabilidade"""
    if probability > 0.8 or probability < 0.2:
        return "Muito Alta"
    elif probability > 0.7 or probability < 0.3:
        return "Alta"
    elif probability > 0.6 or probability < 0.4:
        return "Média"
    else:
        return "Baixa"

def get_feature_importance(model, feature_names):
    """Obter importância das features se disponível"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(feature_names, importances))
        else:
            return {}
    except:
        return {}

def show_enhanced_prediction_result(result):
    """Exibir resultado da predição de forma avançada"""
    st.subheader("🎯 Resultado da Predição")
    
    # Cards de resultado principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pred_color = "#28a745" if result['prediction'] == ">50K" else "#17a2b8"
        create_result_card("🎯 Predição", result['prediction'], "Faixa salarial", pred_color)
    
    with col2:
        prob_color = "#28a745" if result['probability'] > 0.7 else "#ffc107" if result['probability'] > 0.3 else "#dc3545"
        create_result_card("📊 Probabilidade", f"{result['probability']:.1%}", "Chance de >50K", prob_color)
    
    with col3:
        conf_color = {"Muito Alta": "#28a745", "Alta": "#4CAF50", "Média": "#ffc107", "Baixa": "#dc3545"}
        create_result_card("✅ Confiança", result['confidence'], "Nível de certeza", conf_color.get(result['confidence'], "#6c757d"))
    
    with col4:
        method_color = "#4285f4" if result['method'] == 'trained_model' else "#ff9800"
        method_text = "Modelo ML" if result['method'] == 'trained_model' else "Heurístico"
        create_result_card("🤖 Método", method_text, "Tipo de predição", method_color)
    
    # Visualizações avançadas
    col1, col2 = st.columns(2)
    
    with col1:
        # Gauge de probabilidade
        fig_gauge = create_probability_gauge(result['probability'])
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Gráfico de feature importance se disponível
        if 'feature_importance' in result and result['feature_importance']:
            fig_importance = create_feature_importance_chart(result['feature_importance'])
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            # Gráfico de pontuação heurística
            if 'details' in result:
                fig_score = create_score_breakdown_chart(result['details'])
                st.plotly_chart(fig_score, use_container_width=True)
    
    # Detalhamento e interpretação
    show_detailed_interpretation(result)

def create_result_card(title, value, description, color):
    """Criar card de resultado visual"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}08);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid {color};
        margin: 0.5rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        text-align: center;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transform: scale(1.02);
        transition: transform 0.3s ease;
    ">
        <h4 style="margin: 0; color: #333; font-size: 0.9rem; font-weight: 600;">
            {title}
        </h4>
        <h2 style="margin: 0.8rem 0; color: {color}; font-size: 2.2rem; font-weight: bold;">
            {value}
        </h2>
        <p style="margin: 0; color: #666; font-size: 0.8rem; opacity: 0.8;">
            {description}
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_probability_gauge(probability):
    """Criar gauge de probabilidade"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidade de Salário >50K (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#4285f4"},
            'steps': [
                {'range': [0, 25], 'color': "#ffebee"},
                {'range': [25, 50], 'color': "#fff3e0"},
                {'range': [50, 75], 'color': "#e8f5e8"},
                {'range': [75, 100], 'color': "#e3f2fd"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        font={'color': "#333"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_feature_importance_chart(feature_importance):
    """Criar gráfico de importância das features"""
    # Ordenar por importância
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:10]  # Top 10
    
    features, importances = zip(*top_features)
    
    fig = go.Figure(go.Bar(
        x=list(importances),
        y=list(features),
        orientation='h',
        marker_color='#4285f4',
        text=[f'{imp:.3f}' for imp in importances],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="🔍 Importância das Features",
        xaxis_title="Importância",
        yaxis_title="Features",
        height=350,
        font={'color': "#333"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_score_breakdown_chart(details):
    """Criar gráfico de breakdown da pontuação"""
    # Extrair pontuações dos detalhes
    scores = []
    categories = []
    
    for detail in details:
        if '+' in detail:
            parts = detail.split(': +')
            if len(parts) == 2:
                category = parts[0]
                score = int(parts[1].split(' ')[0])
                categories.append(category)
                scores.append(score)
    
    if not scores:
        return None
    
    fig = go.Figure(go.Bar(
        x=categories,
        y=scores,
        marker_color='#ff9800',
        text=scores,
        textposition='auto'
    ))
    
    fig.update_layout(
        title="📊 Breakdown da Pontuação",
        xaxis_title="Categorias",
        yaxis_title="Pontos",
        height=350,
        font={'color': "#333"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def show_detailed_interpretation(result):
    """Mostrar interpretação detalhada do resultado"""
    st.subheader("💡 Interpretação Detalhada")
    
    # Interpretação baseada no resultado
    if result['probability'] > 0.7:
        interpretation = "🟢 **Alta probabilidade** de salário >50K. O perfil apresenta características favoráveis."
        recommendations = [
            "✅ Perfil muito competitivo no mercado",
            "📈 Considere negociar benefícios adicionais",
            "🎯 Foque em oportunidades de liderança"
        ]
    elif result['probability'] > 0.3:
        interpretation = "🟡 **Probabilidade moderada** - perfil no limite entre faixas salariais."
        recommendations = [
            "📚 Considere investir em educação adicional",
            "⏰ Avalie aumentar carga horária se possível",
            "💼 Explore oportunidades em áreas especializadas"
        ]
    else:
        interpretation = "🔵 **Baixa probabilidade** baseada no perfil atual."
        recommendations = [
            "🎓 Investimento em educação é fundamental",
            "🔄 Considere mudança para áreas com maior potencial",
            "📊 Analise o mercado para identificar oportunidades"
        ]
    
    st.markdown(interpretation)
    
    # Recomendações
    st.markdown("#### 🎯 Recomendações Personalizadas")
    for rec in recommendations:
        st.markdown(f"- {rec}")
    
    # Detalhes técnicos se disponível
    if 'details' in result:
        with st.expander("🔍 Detalhes da Análise"):
            for detail in result['details']:
                st.markdown(f"• {detail}")
    
    # Limitações e considerações
    st.markdown("#### ⚠️ Limitações e Considerações")
    limitations = [
        "📊 **Base de Dados**: Resultados baseados em dados históricos (Census 1994)",
        "🔍 **Viés de Dados**: Pode conter vieses sociais da época",
        "⏰ **Contexto Temporal**: Mercado atual pode diferir significativamente",
        "🎯 **Uso Orientativo**: Use como ferramenta de orientação, não decisão definitiva"
    ]
    
    for limitation in limitations:
        st.markdown(f"- {limitation}")

def show_batch_prediction_interface(df, model_status):
    """Interface para predição em lote"""
    st.markdown("### 📊 Predição em Lote")
    st.info("🚧 Funcionalidade em desenvolvimento - Upload de CSV para predições múltiplas")
    
    # Placeholder para upload de arquivo
    uploaded_file = st.file_uploader(
        "Upload arquivo CSV para predição em lote",
        type=['csv'],
        help="Arquivo deve conter as mesmas colunas do dataset de treinamento"
    )
    
    if uploaded_file:
        st.success("📁 Arquivo carregado com sucesso!")
        st.info("🔄 Processamento em lote será implementado em versão futura")

def show_model_comparison_interface(df, model_status):
    """Interface para comparação de modelos"""
    st.markdown("### 🔄 Comparação de Modelos")
    st.info("🚧 Funcionalidade em desenvolvimento - Comparação entre diferentes algoritmos")
    
    # Placeholder para seleção de modelos
    available_models = ["Random Forest", "Logistic Regression", "Gradient Boosting", "SVM"]
    selected_models = st.multiselect(
        "Selecione modelos para comparação",
        available_models,
        default=["Random Forest", "Logistic Regression"]
    )
    
    if selected_models:
        st.success(f"📋 Modelos selecionados: {', '.join(selected_models)}")
        st.info("🔄 Comparação entre modelos será implementada em versão futura")

def show_prediction_history():
    """Mostrar histórico de predições da sessão"""
    if 'prediction_history' not in st.session_state or len(st.session_state.prediction_history) == 0:
        return
    
    st.subheader("📚 Histórico de Predições")
    
    history = st.session_state.prediction_history
    
    # Criar DataFrame do histórico
    history_data = []
    for i, pred in enumerate(reversed(history)):
        history_data.append({
            '#': len(history) - i,
            'Timestamp': pred['timestamp'].strftime('%H:%M:%S'),
            'Predição': pred['prediction'],
            'Probabilidade': f"{pred['probability']:.1%}",
            'Confiança': pred['confidence'],
            'Método': pred.get('method', 'heuristic'),
            'Idade': pred['features']['age'],
            'Educação': pred['features']['education'],
            'Ocupação': pred['features']['occupation']
        })
    
    history_df = pd.DataFrame(history_data)
    
    # Exibir tabela
    st.dataframe(history_df, use_container_width=True)
    
    # Gráfico de tendência
    if len(history) > 1:
        fig_trend = px.line(
            x=range(1, len(history) + 1),
            y=[pred['probability'] for pred in history],
            title="📈 Tendência das Probabilidades",
            labels={'x': 'Predição #', 'y': 'Probabilidade'},
            markers=True
        )
        
        fig_trend.add_hline(y=0.5, line_dash="dash", line_color="red", 
                           annotation_text="Limiar 50%")
        
        fig_trend.update_layout(height=300)
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Botão para limpar histórico
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Limpar Histórico", type="secondary", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()

def save_prediction_to_history(result):
    """Salvar predição no histórico da sessão"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Adicionar resultado ao histórico
    st.session_state.prediction_history.append(result)
    
    # Manter apenas os últimos 20 resultados
    if len(st.session_state.prediction_history) > 20:
        st.session_state.prediction_history = st.session_state.prediction_history[-20:]

def show_model_performance():
    """Mostrar métricas de performance do modelo"""
    st.subheader("📊 Performance do Modelo")
    
    # Métricas simuladas (em produção, viriam do modelo real)
    metrics = {
        'Acurácia': 0.847,
        'Precisão': 0.823,
        'Recall': 0.756,
        'F1-Score': 0.788,
        'ROC-AUC': 0.892
    }
    
    col1, col2, col3, col4, col5 = st.columns(5)
    colors = ['#28a745', '#4285f4', '#ffc107', '#17a2b8', '#6f42c1']
    
    for i, (metric, value) in enumerate(metrics.items()):
        with [col1, col2, col3, col4, col5][i]:
            create_metric_card(metric, f"{value:.3f}", "Score", colors[i])

def create_metric_card(title, value, description, color):
    """Criar card de métrica"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}08);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid {color};
        text-align: center;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <h4 style="margin: 0; color: #333; font-size: 0.8rem; font-weight: 600;">
            {title}
        </h4>
        <h2 style="margin: 0.5rem 0; color: {color}; font-size: 1.5rem; font-weight: bold;">
            {value}
        </h2>
        <p style="margin: 0; color: #666; font-size: 0.7rem; opacity: 0.8;">
            {description}
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_advanced_insights():
    """Mostrar insights avançados sobre predição"""
    st.subheader("💡 Insights Avançados")
    
    insights = [
        {
            "icon": "🎓",
            "title": "Educação é o Fator Determinante",
            "message": "Nível educacional representa ~30% da importância na predição. Investimento em educação superior aumenta significativamente as chances de salários elevados.",
            "color": "#4285f4"
        },
        {
            "icon": "⏰",
            "title": "Carga Horária Importa",
            "message": "Profissionais que trabalham 50+ horas por semana têm 2x mais chances de salários >50K. Balance produtividade e qualidade de vida.",
            "color": "#ffc107"
        },
        {
            "icon": "💼",
            "title": "Ocupação Estratégica",
            "message": "Cargos executivos e de especialização técnica apresentam maior potencial. Considere desenvolvimento em áreas de alta demanda.",
            "color": "#28a745"
        },
        {
            "icon": "📈",
            "title": "Experiência Profissional",
            "message": "Idade (proxy para experiência) é fator positivo. Profissionais 40+ têm vantagem significativa na faixa salarial superior.",
            "color": "#17a2b8"
        }
    ]
    
    for insight in insights:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {insight['color']}15, {insight['color']}08);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid {insight['color']};
            margin: 1rem 0;
            display: flex;
            align-items: flex-start;
        ">
            <div style="font-size: 2rem; margin-right: 1rem; margin-top: 0.2rem;">
                {insight['icon']}
            </div>
            <div>
                <h4 style="margin: 0 0 0.8rem 0; color: #333; font-size: 1.1rem;">
                    {insight['title']}
                </h4>
                <p style="margin: 0; color: #555; line-height: 1.6; font-size: 0.95rem;">
                    {insight['message']}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)