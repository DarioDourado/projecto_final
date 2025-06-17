"""
Página de Modelos de Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def show_models_page(data, i18n):
    """
    Mostrar página de modelos de ML
    """
    st.header("🤖 Modelos de Machine Learning")
    
    if data is None or data.empty:
        st.warning("⚠️ Dados não carregados. Verifique a conexão com a fonte de dados.")
        return
    
    # Sidebar para opções
    st.sidebar.header("🔧 Configurações do Modelo")
    
    model_action = st.sidebar.selectbox(
        "Ação:",
        ["Treinar Novo Modelo", "Carregar Modelo Existente", "Comparar Modelos"]
    )
    
    if model_action == "Treinar Novo Modelo":
        show_train_model(data)
    elif model_action == "Carregar Modelo Existente":
        show_load_model(data)
    elif model_action == "Comparar Modelos":
        show_compare_models(data)

def show_train_model(data):
    """Interface para treinar novo modelo"""
    st.subheader("🏋️ Treinar Novo Modelo")
    
    # Verificar se há dados numéricos suficientes
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Necessário pelo menos 2 variáveis numéricas para treinar um modelo")
        return
    
    # Seleção de variáveis
    col1, col2 = st.columns(2)
    
    with col1:
        target_var = st.selectbox("Variável Target (y):", numeric_cols)
    
    with col2:
        feature_vars = st.multiselect(
            "Variáveis Features (X):", 
            [col for col in numeric_cols if col != target_var],
            default=[col for col in numeric_cols if col != target_var][:3]
        )
    
    if not feature_vars:
        st.warning("Selecione pelo menos uma variável feature")
        return
    
    # Parâmetros do modelo
    st.subheader("⚙️ Parâmetros do Modelo")
    
    model_type = st.selectbox(
        "Tipo de Modelo:",
        ["Random Forest", "Regressão Linear"]
    )
    
    test_size = st.slider("Tamanho do conjunto de teste:", 0.1, 0.5, 0.2, 0.05)
    
    if st.button("🚀 Treinar Modelo", type="primary"):
        train_and_evaluate_model(data, target_var, feature_vars, model_type, test_size)

def train_and_evaluate_model(data, target_var, feature_vars, model_type, test_size):
    """Treinar e avaliar modelo"""
    try:
        # Preparar dados
        X = data[feature_vars].dropna()
        y = data.loc[X.index, target_var]
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Treinar modelo
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        
        with st.spinner("Treinando modelo..."):
            model.fit(X_train, y_train)
        
        # Fazer predições
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Avaliar modelo
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Mostrar resultados
        st.success("✅ Modelo treinado com sucesso!")
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R² (Treino)", f"{train_r2:.3f}")
            st.metric("R² (Teste)", f"{test_r2:.3f}")
        
        with col2:
            st.metric("RMSE (Treino)", f"{train_rmse:.2f}")
            st.metric("RMSE (Teste)", f"{test_rmse:.2f}")
        
        with col3:
            st.metric("MAE (Treino)", f"{train_mae:.2f}")
            st.metric("MAE (Teste)", f"{test_mae:.2f}")
        
        # Gráficos de avaliação
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot: Real vs Predito
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=y_test, 
                y=y_pred_test,
                mode='markers',
                name='Teste',
                opacity=0.7
            ))
            fig_scatter.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='Linha Perfeita',
                line=dict(dash='dash', color='red')
            ))
            fig_scatter.update_layout(
                title="Real vs Predito",
                xaxis_title="Valor Real",
                yaxis_title="Valor Predito"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Resíduos
            residuals = y_test - y_pred_test
            fig_residuals = px.scatter(
                x=y_pred_test,
                y=residuals,
                title="Gráfico de Resíduos",
                labels={'x': 'Valores Preditos', 'y': 'Resíduos'}
            )
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Importância das features (para Random Forest)
        if model_type == "Random Forest":
            st.subheader("📊 Importância das Features")
            feature_importance = pd.DataFrame({
                'Feature': feature_vars,
                'Importância': model.feature_importances_
            }).sort_values('Importância', ascending=False)
            
            fig_importance = px.bar(
                feature_importance,
                x='Importância',
                y='Feature',
                orientation='h',
                title="Importância das Features"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Salvar modelo
        if st.button("💾 Salvar Modelo"):
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            model_filename = f"{model_type.lower().replace(' ', '_')}_{target_var}_{len(feature_vars)}features.pkl"
            model_path = models_dir / model_filename
            
            joblib.dump({
                'model': model,
                'target_var': target_var,
                'feature_vars': feature_vars,
                'model_type': model_type,
                'metrics': {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse
                }
            }, model_path)
            
            st.success(f"✅ Modelo salvo em: {model_path}")
    
    except Exception as e:
        st.error(f"❌ Erro ao treinar modelo: {e}")

def show_load_model(data):
    """Interface para carregar modelo existente"""
    st.subheader("📂 Carregar Modelo Existente")
    
    models_dir = Path("models")
    if not models_dir.exists():
        st.warning("Diretório 'models' não encontrado")
        return
    
    model_files = list(models_dir.glob("*.pkl"))
    if not model_files:
        st.warning("Nenhum modelo encontrado no diretório")
        return
    
    selected_model = st.selectbox(
        "Selecione o modelo:",
        model_files,
        format_func=lambda x: x.name
    )
    
    if st.button("📁 Carregar Modelo"):
        try:
            model_data = joblib.load(selected_model)
            
            st.success("✅ Modelo carregado com sucesso!")
            
            # Mostrar informações do modelo
            st.json({
                'Tipo': model_data['model_type'],
                'Target': model_data['target_var'],
                'Features': model_data['feature_vars'],
                'Métricas': model_data['metrics']
            })
            
            # Interface de predição
            st.subheader("🔮 Fazer Predições")
            
            # Criar inputs para cada feature
            feature_values = {}
            for feature in model_data['feature_vars']:
                if feature in data.columns:
                    min_val = float(data[feature].min())
                    max_val = float(data[feature].max())
                    mean_val = float(data[feature].mean())
                    
                    feature_values[feature] = st.number_input(
                        f"{feature}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"pred_{feature}"
                    )
            
            if st.button("🎯 Fazer Predição"):
                input_data = pd.DataFrame([feature_values])
                prediction = model_data['model'].predict(input_data)[0]
                
                st.success(f"📊 Predição: {prediction:.2f}")
        
        except Exception as e:
            st.error(f"❌ Erro ao carregar modelo: {e}")

def show_compare_models(data):
    """Interface para comparar modelos"""
    st.subheader("⚖️ Comparar Modelos")
    
    models_dir = Path("models")
    if not models_dir.exists() or not list(models_dir.glob("*.pkl")):
        st.warning("Nenhum modelo encontrado para comparação")
        return
    
    model_files = list(models_dir.glob("*.pkl"))
    
    # Carregar métricas de todos os modelos
    models_comparison = []
    
    for model_file in model_files:
        try:
            model_data = joblib.load(model_file)
            models_comparison.append({
                'Nome': model_file.name,
                'Tipo': model_data['model_type'],
                'Target': model_data['target_var'],
                'Features': len(model_data['feature_vars']),
                'R² Teste': model_data['metrics']['test_r2'],
                'RMSE Teste': model_data['metrics']['test_rmse']
            })
        except Exception as e:
            st.warning(f"Erro ao carregar {model_file.name}: {e}")
    
    if models_comparison:
        comparison_df = pd.DataFrame(models_comparison)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Gráfico de comparação
        fig_comparison = px.scatter(
            comparison_df,
            x='RMSE Teste',
            y='R² Teste',
            size='Features',
            color='Tipo',
            title="Comparação de Modelos",
            hover_data=['Nome']
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    else:
        st.warning("Nenhum modelo válido encontrado")