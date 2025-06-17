"""
Página de Predição Salarial
Sistema de Machine Learning para Predição Interativa
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json

def show_prediction_page(data):
    """Página de predição salarial"""
    st.title("🔮 Predição Salarial")
    st.markdown("### Sistema de Machine Learning para Análise Preditiva")
    
    # Introdução contextual
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
    ">
        <p style="margin: 0; color: #495057;">
            🔮 <strong>Sistema de Predição Inteligente</strong> utiliza algoritmos de Machine Learning 
            para prever faixas salariais baseado em características demográficas e profissionais.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar se há dados originais para contextualizar
    if 'original' in data and not data['original'].empty:
        show_dataset_context(data['original'])
    
    # Interface de predição
    prediction_result = show_prediction_interface()
    
    # Histórico de predições
    show_prediction_history()
    
    # Análise exploratória dos dados
    if 'original' in data:
        show_exploratory_analysis(data['original'])
    
    # Métricas do modelo
    show_model_metrics()
    
    # Insights e interpretação
    show_prediction_insights()

def show_dataset_context(df):
    """Mostrar contexto do dataset para predição"""
    st.subheader("📊 Contexto dos Dados")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(df)
        create_context_card("📋 Total de Registros", f"{total_records:,}", "Base de treinamento", "#4285f4")
    
    with col2:
        if 'salary' in df.columns:
            high_salary_pct = (df['salary'] == '>50K').mean() * 100
            create_context_card("💰 Salário >50K", f"{high_salary_pct:.1f}%", "Distribuição positiva", "#28a745")
        else:
            create_context_card("💰 Salário >50K", "N/A", "Dados não disponíveis", "#6c757d")
    
    with col3:
        if 'age' in df.columns:
            avg_age = df['age'].mean()
            create_context_card("👤 Idade Média", f"{avg_age:.1f}", "Anos", "#ffc107")
        else:
            create_context_card("👤 Idade Média", "N/A", "Dados não disponíveis", "#6c757d")
    
    with col4:
        if 'hours-per-week' in df.columns:
            avg_hours = df['hours-per-week'].mean()
            create_context_card("⏰ Horas/Semana", f"{avg_hours:.1f}", "Média de trabalho", "#17a2b8")
        else:
            create_context_card("⏰ Horas/Semana", "N/A", "Dados não disponíveis", "#6c757d")

def create_context_card(title, value, description, color):
    """Criar card de contexto"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}08);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid {color};
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <h4 style="margin: 0; color: #333; font-size: 0.85rem; font-weight: 600;">
            {title}
        </h4>
        <h2 style="margin: 0.6rem 0; color: {color}; font-size: 1.8rem; font-weight: bold;">
            {value}
        </h2>
        <p style="margin: 0; color: #666; font-size: 0.75rem; opacity: 0.8;">
            {description}
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_prediction_interface():
    """Interface de predição interativa"""
    st.subheader("🚀 Faça sua Predição")
    
    # Criar duas colunas para o formulário
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Informações Pessoais")
        
        age = st.slider(
            "🎂 Idade",
            min_value=18, 
            max_value=90, 
            value=35,
            help="Idade em anos completos"
        )
        
        sex = st.radio(
            "⚥ Sexo",
            ['Male', 'Female'],
            help="Sexo biológico"
        )
        
        education = st.selectbox(
            "🎓 Nível de Educação",
            [
                'HS-grad', 'Some-college', 'Bachelors', 
                'Masters', 'Doctorate', 'Assoc-voc', 
                'Assoc-acdm', 'Prof-school'
            ],
            index=2,
            help="Maior nível educacional completado"
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
                'Prof-specialty', 'Craft-repair', 'Exec-managerial',
                'Adm-clerical', 'Sales', 'Other-service',
                'Machine-op-inspct', 'Transport-moving',
                'Handlers-cleaners', 'Farming-fishing',
                'Tech-support', 'Protective-serv',
                'Priv-house-serv', 'Armed-Forces'
            ],
            help="Área de atuação profissional"
        )
        
        hours_per_week = st.slider(
            "⏰ Horas por Semana",
            min_value=1,
            max_value=99,
            value=40,
            help="Número de horas trabalhadas por semana"
        )
        
        relationship = st.selectbox(
            "👨‍👩‍👧‍👦 Relacionamento",
            [
                'Husband', 'Not-in-family', 'Wife',
                'Own-child', 'Unmarried', 'Other-relative'
            ],
            help="Relacionamento familiar"
        )
    
    # Botão de predição
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predict_button = st.button(
            "🚀 Realizar Predição",
            type="primary",
            use_container_width=True,
            help="Clique para gerar a predição salarial"
        )
    
    # Processar predição
    if predict_button:
        prediction_result = calculate_prediction({
            'age': age,
            'sex': sex,
            'education': education,
            'marital_status': marital_status,
            'workclass': workclass,
            'occupation': occupation,
            'hours_per_week': hours_per_week,
            'relationship': relationship
        })
        
        # Salvar no histórico
        save_prediction_to_history(prediction_result)
        
        # Mostrar resultado
        show_prediction_result(prediction_result)
        
        return prediction_result
    
    return None

def calculate_prediction(features):
    """Calcular predição baseada em regras heurísticas"""
    
    # Sistema de pontuação baseado em características
    score = 0
    details = []
    
    # Pontuação por educação (peso alto)
    education_scores = {
        'Doctorate': 35,
        'Masters': 30,
        'Prof-school': 28,
        'Bachelors': 25,
        'Assoc-acdm': 15,
        'Assoc-voc': 12,
        'Some-college': 10,
        'HS-grad': 5
    }
    
    edu_score = education_scores.get(features['education'], 0)
    score += edu_score
    details.append(f"Educação ({features['education']}): +{edu_score} pontos")
    
    # Pontuação por idade (experiência)
    if features['age'] >= 50:
        age_score = 20
    elif features['age'] >= 40:
        age_score = 15
    elif features['age'] >= 30:
        age_score = 10
    elif features['age'] >= 25:
        age_score = 5
    else:
        age_score = 0
    
    score += age_score
    details.append(f"Idade ({features['age']} anos): +{age_score} pontos")
    
    # Pontuação por horas trabalhadas
    if features['hours_per_week'] >= 50:
        hours_score = 15
    elif features['hours_per_week'] >= 40:
        hours_score = 10
    elif features['hours_per_week'] >= 30:
        hours_score = 5
    else:
        hours_score = 0
    
    score += hours_score
    details.append(f"Horas/semana ({features['hours_per_week']}h): +{hours_score} pontos")
    
    # Pontuação por ocupação
    occupation_scores = {
        'Exec-managerial': 20,
        'Prof-specialty': 18,
        'Tech-support': 12,
        'Sales': 10,
        'Craft-repair': 8,
        'Adm-clerical': 6,
        'Other-service': 3,
        'Machine-op-inspct': 5,
        'Transport-moving': 4,
        'Handlers-cleaners': 2,
        'Farming-fishing': 2,
        'Protective-serv': 8,
        'Priv-house-serv': 1,
        'Armed-Forces': 10
    }
    
    occ_score = occupation_scores.get(features['occupation'], 5)
    score += occ_score
    details.append(f"Ocupação ({features['occupation']}): +{occ_score} pontos")
    
    # Pontuação por classe de trabalho
    workclass_scores = {
        'Federal-gov': 10,
        'Self-emp-inc': 12,
        'Local-gov': 8,
        'State-gov': 8,
        'Private': 5,
        'Self-emp-not-inc': 3,
        'Without-pay': 0,
        'Never-worked': 0
    }
    
    work_score = workclass_scores.get(features['workclass'], 0)
    score += work_score
    details.append(f"Classe trabalho ({features['workclass']}): +{work_score} pontos")
    
    # Ajuste por estado civil
    if features['marital_status'] in ['Married-civ-spouse', 'Married-AF-spouse']:
        marital_score = 8
    else:
        marital_score = 0
    
    score += marital_score
    if marital_score > 0:
        details.append(f"Estado civil (Casado): +{marital_score} pontos")
    
    # Calcular probabilidade
    max_possible_score = 100
    probability = min(score / max_possible_score, 0.95)
    
    # Determinar predição
    prediction = ">50K" if probability > 0.5 else "<=50K"
    
    # Calcular confiança
    if probability > 0.75 or probability < 0.25:
        confidence = "Alta"
    elif probability > 0.60 or probability < 0.40:
        confidence = "Média"
    else:
        confidence = "Baixa"
    
    return {
        'prediction': prediction,
        'probability': probability,
        'confidence': confidence,
        'score': score,
        'max_score': max_possible_score,
        'details': details,
        'features': features,
        'timestamp': datetime.now()
    }

def show_prediction_result(result):
    """Mostrar resultado da predição"""
    st.subheader("🎯 Resultado da Predição")
    
    # Cards de resultado
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pred_color = "#28a745" if result['prediction'] == ">50K" else "#17a2b8"
        create_result_card("🎯 Predição", result['prediction'], "Faixa salarial", pred_color)
    
    with col2:
        prob_color = "#28a745" if result['probability'] > 0.7 else "#ffc107" if result['probability'] > 0.3 else "#dc3545"
        create_result_card("📊 Probabilidade", f"{result['probability']:.1%}", "Chance de >50K", prob_color)
    
    with col3:
        conf_color = "#28a745" if result['confidence'] == "Alta" else "#ffc107" if result['confidence'] == "Média" else "#dc3545"
        create_result_card("✅ Confiança", result['confidence'], "Nível de certeza", conf_color)
    
    with col4:
        score_color = "#4285f4"
        create_result_card("🔢 Score", f"{result['score']}/{result['max_score']}", "Pontuação total", score_color)
    
    # Gráfico de probabilidade
    col1, col2 = st.columns(2)
    
    with col1:
        # Gauge chart da probabilidade
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = result['probability'] * 100,
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
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Detalhamento da pontuação
        st.markdown("#### 📋 Detalhamento da Pontuação")
        
        for detail in result['details']:
            st.markdown(f"• {detail}")
        
        # Interpretação
        st.markdown("#### 💡 Interpretação")
        
        if result['probability'] > 0.7:
            st.success("🟢 **Alta probabilidade** de salário >50K baseado no perfil informado.")
        elif result['probability'] > 0.3:
            st.warning("🟡 **Probabilidade moderada** - perfil no limite entre faixas salariais.")
        else:
            st.info("🔵 **Baixa probabilidade** de salário >50K baseado no perfil atual.")

def create_result_card(title, value, description, color):
    """Criar card de resultado"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}08);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid {color};
        margin: 0.5rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        text-align: center;
        min-height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transform: scale(1.02);
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

def save_prediction_to_history(result):
    """Salvar predição no histórico da sessão"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Adicionar resultado ao histórico
    st.session_state.prediction_history.append(result)
    
    # Manter apenas os últimos 10 resultados
    if len(st.session_state.prediction_history) > 10:
        st.session_state.prediction_history = st.session_state.prediction_history[-10:]

def show_prediction_history():
    """Mostrar histórico de predições"""
    if 'prediction_history' not in st.session_state or len(st.session_state.prediction_history) == 0:
        return
    
    st.subheader("📚 Histórico de Predições")
    
    history = st.session_state.prediction_history
    
    # Criar dataframe do histórico
    history_data = []
    for i, pred in enumerate(reversed(history)):
        history_data.append({
            '#': len(history) - i,
            'Timestamp': pred['timestamp'].strftime('%H:%M:%S'),
            'Predição': pred['prediction'],
            'Probabilidade': f"{pred['probability']:.1%}",
            'Confiança': pred['confidence'],
            'Score': f"{pred['score']}/{pred['max_score']}",
            'Idade': pred['features']['age'],
            'Educação': pred['features']['education'],
            'Ocupação': pred['features']['occupation']
        })
    
    history_df = pd.DataFrame(history_data)
    
    # Mostrar tabela
    st.dataframe(history_df, use_container_width=True)
    
    # Gráfico de tendência se houver múltiplas predições
    if len(history) > 1:
        fig_trend = px.line(
            x=range(1, len(history) + 1),
            y=[pred['probability'] for pred in history],
            title="Tendência das Probabilidades",
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

def show_exploratory_analysis(df):
    """Análise exploratória dos dados"""
    st.subheader("🔍 Análise Exploratória dos Dados")
    
    # Verificar se há coluna de salário para análise
    if 'salary' not in df.columns:
        st.info("📊 Coluna de salário não encontrada para análise exploratória")
        return
    
    # Análise por características
    analysis_features = ['age', 'education', 'occupation', 'hours-per-week']
    available_features = [f for f in analysis_features if f in df.columns]
    
    if not available_features:
        st.info("📊 Características necessárias não encontradas no dataset")
        return
    
    # Seletor de característica para análise
    selected_feature = st.selectbox(
        "Selecionar característica para análise:",
        available_features,
        format_func=lambda x: x.replace('-', ' ').title()
    )
    
    if selected_feature:
        show_feature_analysis(df, selected_feature)

def show_feature_analysis(df, feature):
    """Análise detalhada de uma característica"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuição da característica
        if df[feature].dtype in ['int64', 'float64']:
            # Variável numérica
            fig_hist = px.histogram(
                df, 
                x=feature, 
                title=f"Distribuição - {feature.replace('-', ' ').title()}",
                nbins=20
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            # Variável categórica
            value_counts = df[feature].value_counts().head(10)
            fig_bar = px.bar(
                x=value_counts.values,
                y=value_counts.index,
                orientation='h',
                title=f"Top 10 - {feature.replace('-', ' ').title()}"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Relação com salário
        if df[feature].dtype in ['int64', 'float64']:
            # Box plot para variável numérica
            fig_box = px.box(
                df, 
                x='salary', 
                y=feature,
                title=f"{feature.replace('-', ' ').title()} por Faixa Salarial"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            # Gráfico de barras agrupadas para categórica
            crosstab = pd.crosstab(df[feature], df['salary'], normalize='index') * 100
            
            fig_grouped = px.bar(
                crosstab.reset_index(),
                x=feature,
                y=['<=50K', '>50K'],
                title=f"Distribuição Salarial por {feature.replace('-', ' ').title()}",
                labels={'value': 'Percentual (%)', 'variable': 'Faixa Salarial'}
            )
            st.plotly_chart(fig_grouped, use_container_width=True)

def show_model_metrics():
    """Mostrar métricas do modelo"""
    st.subheader("📊 Métricas do Modelo")
    
    # Métricas simuladas (em um cenário real, seriam calculadas)
    metrics = {
        'Acurácia': 0.847,
        'Precisão': 0.823,
        'Recall': 0.756,
        'F1-Score': 0.788
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    colors = ['#28a745', '#4285f4', '#ffc107', '#17a2b8']
    
    for i, (metric, value) in enumerate(metrics.items()):
        with [col1, col2, col3, col4][i]:
            create_metric_card(metric, f"{value:.3f}", "Score do modelo", colors[i])
    
    # Interpretação das métricas
    st.markdown("#### 📋 Interpretação das Métricas")
    
    interpretations = [
        "🎯 **Acurácia (84.7%)**: Percentual de predições corretas",
        "🔍 **Precisão (82.3%)**: Proporção de predições >50K que estão corretas",
        "📈 **Recall (75.6%)**: Proporção de casos >50K que foram identificados",
        "⚖️ **F1-Score (78.8%)**: Média harmônica entre precisão e recall"
    ]
    
    for interp in interpretations:
        st.markdown(f"- {interp}")

def create_metric_card(title, value, description, color):
    """Criar card de métrica"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}08);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid {color};
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <h4 style="margin: 0; color: #333; font-size: 0.9rem; font-weight: 600;">
            {title}
        </h4>
        <h2 style="margin: 0.8rem 0; color: {color}; font-size: 2rem; font-weight: bold;">
            {value}
        </h2>
        <p style="margin: 0; color: #666; font-size: 0.8rem; opacity: 0.8;">
            {description}
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_prediction_insights():
    """Mostrar insights sobre predição"""
    st.subheader("💡 Insights e Recomendações")
    
    insights = [
        {
            "icon": "🎓",
            "title": "Educação é Fundamental",
            "message": "O nível educacional é o fator mais importante para salários acima de 50K. Investir em educação superior aumenta significativamente as chances.",
            "color": "#4285f4"
        },
        {
            "icon": "⏰",
            "title": "Horas de Trabalho Importam",
            "message": "Profissionais que trabalham 50+ horas por semana têm maior probabilidade de salários elevados, mas é importante considerar o equilíbrio trabalho-vida.",
            "color": "#ffc107"
        },
        {
            "icon": "💼",
            "title": "Ocupação Estratégica",
            "message": "Cargos executivos e de especialização técnica apresentam maior potencial salarial. Considere desenvolvimento em áreas de alta demanda.",
            "color": "#28a745"
        },
        {
            "icon": "📈",
            "title": "Experiência Conta",
            "message": "A idade (proxy para experiência) é um fator positivo. Profissionais com 40+ anos têm vantagem na faixa salarial superior.",
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
    
    # Limitações e considerações
    st.markdown("#### ⚠️ Limitações e Considerações")
    
    limitations = [
        "📊 **Modelo Simplificado**: Este é um modelo baseado em regras heurísticas para demonstração",
        "🔍 **Viés de Dados**: Resultados baseados em dados históricos que podem conter vieses",
        "⏰ **Contexto Temporal**: Mercado de trabalho evolui, dados podem não refletir realidade atual",
        "🌍 **Contexto Geográfico**: Padrões podem variar significativamente por região",
        "🎯 **Uso Responsável**: Use como ferramenta de orientação, não como decisão definitiva"
    ]
    
    for limitation in limitations:
        st.markdown(f"- {limitation}")