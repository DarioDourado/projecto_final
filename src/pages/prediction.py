"""
🔮 Página de Predição
Interface para fazer predições individuais e em lote
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import io

def show_prediction_page(data, i18n):
    """Página principal de predição"""
    # Import safe
    try:
        from src.components.navigation import show_page_header
    except ImportError:
        def show_page_header(title, subtitle, icon):
            st.markdown(f"## {icon} {title}")
            if subtitle:
                st.markdown(f"*{subtitle}*")
    
    show_page_header(
        i18n.t('navigation.prediction', 'Predição'),
        i18n.t('prediction.subtitle', 'Fazer predições de salário com novos dados'),
        "🔮"
    )
    
    # Verificar se modelo está disponível
    model_available = _check_model_availability()
    
    if not model_available:
        st.error("❌ Modelo não encontrado! Execute o pipeline primeiro: `python main.py`")
        return
    
    # Tabs para diferentes tipos de predição
    tab1, tab2, tab3 = st.tabs([
        f"👤 {i18n.t('prediction.individual', 'Predição Individual')}",
        f"📊 {i18n.t('prediction.batch', 'Predição em Lote')}",
        f"🎯 {i18n.t('prediction.examples', 'Exemplos')}"
    ])
    
    with tab1:
        _show_individual_prediction(i18n)
    
    with tab2:
        _show_batch_prediction(i18n)
    
    with tab3:
        _show_prediction_examples(i18n)

def _check_model_availability():
    """Verificar se modelo está disponível"""
    model_paths = [
        "random_forest_model.joblib",
        "models/random_forest_model.joblib",
        "output/models/random_forest_model.joblib"
    ]
    
    for path in model_paths:
        if Path(path).exists():
            return True
    
    return False

def _load_model_and_preprocessor():
    """Carregar modelo e preprocessador"""
    try:
        # Tentar diferentes locais
        model_paths = [
            "random_forest_model.joblib",
            "models/random_forest_model.joblib",
            "output/models/random_forest_model.joblib"
        ]
        
        preprocessor_paths = [
            "preprocessor.joblib",
            "models/preprocessor.joblib", 
            "output/models/preprocessor.joblib"
        ]
        
        model = None
        preprocessor = None
        
        # Carregar modelo
        for path in model_paths:
            if Path(path).exists():
                model = joblib.load(path)
                break
        
        # Carregar preprocessador
        for path in preprocessor_paths:
            if Path(path).exists():
                preprocessor = joblib.load(path)
                break
        
        return model, preprocessor
        
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, None

def _show_individual_prediction(i18n):
    """Predição individual"""
    st.subheader(f"👤 {i18n.t('prediction.individual_title', 'Predição Individual')}")
    
    # Formulário para entrada de dados
    with st.form("prediction_form"):
        st.markdown("#### 📝 Preencha os dados para predição:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("🎂 Idade", min_value=18, max_value=100, value=35)
            
            workclass = st.selectbox("💼 Classe de Trabalho", [
                "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                "Local-gov", "State-gov", "Without-pay", "Never-worked"
            ])
            
            education = st.selectbox("🎓 Educação", [
                "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
                "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
                "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
            ])
            
            marital_status = st.selectbox("💑 Estado Civil", [
                "Married-civ-spouse", "Divorced", "Never-married", "Separated",
                "Widowed", "Married-spouse-absent", "Married-AF-spouse"
            ])
            
            occupation = st.selectbox("🏢 Ocupação", [
                "Tech-support", "Craft-repair", "Other-service", "Sales",
                "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                "Transport-moving", "Priv-house-serv", "Protective-serv",
                "Armed-Forces"
            ])
        
        with col2:
            relationship = st.selectbox("👨‍👩‍👧‍👦 Relacionamento", [
                "Wife", "Own-child", "Husband", "Not-in-family",
                "Other-relative", "Unmarried"
            ])
            
            race = st.selectbox("🌍 Raça", [
                "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
                "Other", "Black"
            ])
            
            sex = st.selectbox("👥 Sexo", ["Male", "Female"])
            
            hours_per_week = st.number_input("⏰ Horas por Semana", min_value=1, max_value=100, value=40)
            
            native_country = st.selectbox("🌎 País de Origem", [
                "United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
                "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan",
                "Greece", "South", "China", "Cuba", "Iran", "Honduras",
                "Philippines", "Italy", "Poland", "Jamaica", "Vietnam",
                "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
                "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary",
                "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
                "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
            ])
        
        # Campos numéricos adicionais (com valores padrão)
        col3, col4 = st.columns(2)
        with col3:
            fnlwgt = st.number_input("📊 Final Weight", value=100000)
            education_num = st.number_input("📚 Anos de Educação", min_value=1, max_value=20, value=13)
        
        with col4:
            capital_gain = st.number_input("💰 Ganho de Capital", min_value=0, value=0)
            capital_loss = st.number_input("📉 Perda de Capital", min_value=0, value=0)
        
        # Botão de predição
        submitted = st.form_submit_button("🎯 FAZER PREDIÇÃO", type="primary")
        
        if submitted:
            # Criar DataFrame com os dados
            input_data = pd.DataFrame({
                'age': [age],
                'workclass': [workclass],
                'fnlwgt': [fnlwgt],
                'education': [education],
                'education-num': [education_num],
                'marital-status': [marital_status],
                'occupation': [occupation],
                'relationship': [relationship],
                'race': [race],
                'sex': [sex],
                'capital-gain': [capital_gain],
                'capital-loss': [capital_loss],
                'hours-per-week': [hours_per_week],
                'native-country': [native_country]
            })
            
            # Fazer predição
            _make_prediction(input_data, i18n, show_details=True)

def _make_prediction(input_data, i18n, show_details=False):
    """Fazer predição com os dados fornecidos"""
    try:
        model, preprocessor = _load_model_and_preprocessor()
        
        if model is None or preprocessor is None:
            st.error("❌ Erro ao carregar modelo ou preprocessador")
            return
        
        # Preprocessar dados
        X_processed = preprocessor.transform(input_data)
        
        # Fazer predição
        prediction = model.predict(X_processed)[0]
        prediction_proba = model.predict_proba(X_processed)[0]
        
        # Mostrar resultado
        if prediction == 1 or prediction == '>50K':
            st.success("💰 **PREDIÇÃO: SALÁRIO ALTO (>$50K)**")
            prob_high = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
            st.info(f"🎯 Confiança: {prob_high:.1%}")
        else:
            st.warning("📊 **PREDIÇÃO: SALÁRIO BAIXO (≤$50K)**")
            prob_low = prediction_proba[0] if len(prediction_proba) > 1 else 1 - prediction_proba[0]
            st.info(f"🎯 Confiança: {prob_low:.1%}")
        
        if show_details:
            # Mostrar detalhes da predição
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("📊 Probabilidade ≤$50K", f"{prediction_proba[0]:.1%}")
            
            with col2:
                if len(prediction_proba) > 1:
                    st.metric("💰 Probabilidade >$50K", f"{prediction_proba[1]:.1%}")
            
            # Gráfico de probabilidades
            _plot_prediction_probabilities(prediction_proba)
        
        return prediction, prediction_proba
        
    except Exception as e:
        st.error(f"❌ Erro na predição: {e}")
        return None, None

def _plot_prediction_probabilities(proba):
    """Plotar probabilidades da predição"""
    try:
        import plotly.graph_objects as go
        
        if len(proba) == 2:
            labels = ['≤$50K', '>$50K']
            values = proba
        else:
            labels = ['Prediction']
            values = [proba[0]]
        
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=values,
            marker_color=['#FF6B6B', '#4ECDC4'][:len(labels)],
            text=[f'{v:.1%}' for v in values],
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Probabilidades da Predição",
            yaxis_title="Probabilidade",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erro ao plotar probabilidades: {e}")

def _show_batch_prediction(i18n):
    """Predição em lote"""
    st.subheader(f"📊 {i18n.t('prediction.batch_title', 'Predição em Lote')}")
    
    st.markdown("""
    #### 📝 Como usar:
    1. Prepare um arquivo CSV com as colunas necessárias
    2. Faça upload do arquivo
    3. Clique em "Fazer Predições em Lote"
    4. Baixe o resultado com as predições
    """)
    
    # Template CSV
    st.markdown("#### 📋 Template CSV")
    if st.button("📥 Baixar Template CSV"):
        template_data = {
            'age': [39, 50, 38],
            'workclass': ['State-gov', 'Self-emp-not-inc', 'Private'],
            'fnlwgt': [77516, 83311, 215646],
            'education': ['Bachelors', 'Bachelors', 'HS-grad'],
            'education-num': [13, 13, 9],
            'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced'],
            'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners'],
            'relationship': ['Not-in-family', 'Husband', 'Not-in-family'],
            'race': ['White', 'White', 'White'],
            'sex': ['Male', 'Male', 'Male'],
            'capital-gain': [2174, 0, 0],
            'capital-loss': [0, 0, 0],
            'hours-per-week': [40, 13, 40],
            'native-country': ['United-States', 'United-States', 'United-States']
        }
        
        template_df = pd.DataFrame(template_data)
        csv = template_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download template CSV",
            data=csv,
            file_name="template_predicao.csv",
            mime="text/csv"
        )
    
    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "📤 Faça upload do seu arquivo CSV:",
        type=["csv"],
        help="O arquivo deve conter as mesmas colunas do template"
    )
    
    if uploaded_file is not None:
        # Ler arquivo CSV
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Arquivo carregado com {len(df)} registros")
            
            # Mostrar amostra dos dados
            st.markdown("#### 📊 Amostra dos Dados")
            st.dataframe(df.head(), use_container_width=True)
            
            # Botão para executar predições em lote
            if st.button("🚀 Fazer Predições em Lote", type="primary"):
                _execute_batch_prediction(df, i18n)
        
        except Exception as e:
            st.error(f"❌ Erro ao ler o arquivo: {e}")

def _show_prediction_examples(i18n):
    """Mostrar exemplos de predições"""
    st.subheader(f"🎯 {i18n.t('prediction.examples_title', 'Exemplos de Predições')}")
    
    st.markdown("""
    Veja alguns exemplos de predições feitas com dados de entrada específicos.
    """)
    
    # Exemplo 1
    st.markdown("#### Exemplo 1")
    example_data_1 = {
        'age': 28,
        'workclass': 'Private',
        'fnlwgt': 234721,
        'education': 'Bachelors',
        'education-num': 13,
        'marital-status': 'Never-married',
        'occupation': 'Sales',
        'relationship': 'Not-in-family',
        'race': 'Black',
        'sex': 'Female',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States'
    }
    
    _make_prediction(pd.DataFrame([example_data_1]), i18n)
    
    # Exemplo 2
    st.markdown("#### Exemplo 2")
    example_data_2 = {
        'age': 45,
        'workclass': 'Self-emp-not-inc',
        'fnlwgt': 123456,
        'education': 'Masters',
        'education-num': 14,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 5000,
        'capital-loss': 0,
        'hours-per-week': 50,
        'native-country': 'United-States'
    }
    
    _make_prediction(pd.DataFrame([example_data_2]), i18n)

def _execute_batch_prediction(df, i18n):
    """Executar predições em lote"""
    st.markdown("### 🚀 Resultados das Predições em Lote")
    
    # Carregar modelo e preprocessador
    model, preprocessor = _load_model_and_preprocessor()
    
    if model is None or preprocessor is None:
        st.error("❌ Erro ao carregar modelo ou preprocessador")
        return
    
    # Preprocessar dados
    X_processed = preprocessor.transform(df)
    
    # Fazer predições
    predictions = model.predict(X_processed)
    prediction_proba = model.predict_proba(X_processed)
    
    # Adicionar resultados ao DataFrame
    df['prediction'] = predictions
    df['probability'] = [proba[1] if pred == '>50K' else proba[0] for pred, proba in zip(predictions, prediction_proba)]
    
    # Mostrar resultados
    st.dataframe(df[['prediction', 'probability']], use_container_width=True)
    
    # Download dos resultados
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download dos Resultados (CSV)",
        data=csv,
        file_name="resultados_predicao.csv",
        mime="text/csv"
    )