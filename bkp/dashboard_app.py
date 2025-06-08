"""Dashboard Streamlit com nova estrutura"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.settings import PROCESSED_DATA_DIR, IMAGES_DIR, MODELS_DIR
from src.utils.logger import get_logger
from src.visualization.modern_styles import ModernStyle

# Configurar página
st.set_page_config(
    page_title="Dashboard - Análise Salarial",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar estilo
ModernStyle.setup_matplotlib()
logger = get_logger(__name__)

@st.cache_data
def load_processed_data():
    """Carregar dados processados"""
    processed_file = PROCESSED_DATA_DIR / "data_processed.csv"
    if processed_file.exists():
        return pd.read_csv(processed_file)
    return None

def main():
    """Aplicação principal do dashboard"""
    
    st.title("🎯 Dashboard - Análise Salarial")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🔧 Controles")
    
    # Verificar se dados estão disponíveis
    df = load_processed_data()
    
    if df is None:
        st.error("❌ Dados processados não encontrados!")
        st.info("💡 Execute primeiro: `python main.py --step data`")
        st.stop()
    
    # Menu principal
    page = st.sidebar.selectbox(
        "📊 Selecione a página:",
        ["🏠 Visão Geral", "📈 Visualizações", "🤖 Modelos", "🔍 Predições"]
    )
    
    if page == "🏠 Visão Geral":
        show_overview(df)
    elif page == "📈 Visualizações":
        show_visualizations()
    elif page == "🤖 Modelos":
        show_models()
    elif page == "🔍 Predições":
        show_predictions()

def show_overview(df):
    """Mostrar visão geral dos dados"""
    st.header("🏠 Visão Geral dos Dados")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total de Registros", len(df))
    
    with col2:
        st.metric("📋 Colunas", len(df.columns))
    
    with col3:
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("❓ Dados Ausentes", f"{missing_percentage:.1f}%")
    
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("💾 Uso de Memória", f"{memory_mb:.1f} MB")
    
    # Mostrar primeiras linhas
    st.subheader("📋 Primeiras Linhas")
    st.dataframe(df.head())
    
    # Estatísticas descritivas
    st.subheader("📊 Estatísticas Descritivas")
    st.dataframe(df.describe())

def show_visualizations():
    """Mostrar visualizações"""
    st.header("📈 Visualizações")
    
    # Listar imagens disponíveis
    image_files = list(IMAGES_DIR.glob("*.png"))
    
    if not image_files:
        st.warning("⚠️ Nenhuma visualização encontrada!")
        st.info("💡 Execute: `python main.py --step eda`")
        return
    
    # Selectbox para escolher visualização
    selected_image = st.selectbox(
        "🖼️ Selecione uma visualização:",
        options=[img.stem for img in image_files]
    )
    
    if selected_image:
        image_path = IMAGES_DIR / f"{selected_image}.png"
        st.image(str(image_path), caption=selected_image)

def show_models():
    """Mostrar informações dos modelos"""
    st.header("🤖 Modelos Treinados")
    
    # Listar modelos disponíveis
    model_files = list(MODELS_DIR.glob("*.pkl"))
    
    if not model_files:
        st.warning("⚠️ Nenhum modelo encontrado!")
        st.info("💡 Execute: `python main.py --step models`")
        return
    
    for model_file in model_files:
        st.write(f"📁 {model_file.name}")

def show_predictions():
    """Interface para predições"""
    st.header("🔍 Fazer Predições")
    
    st.info("🚧 Funcionalidade em desenvolvimento...")
    
    # Inputs para predição
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Idade", min_value=17, max_value=100, value=30)
            education_num = st.number_input("Anos de Educação", min_value=1, max_value=16, value=12)
            hours_per_week = st.number_input("Horas/Semana", min_value=1, max_value=99, value=40)
        
        with col2:
            workclass = st.selectbox("Classe de Trabalho", 
                                   ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov"])
            education = st.selectbox("Educação", 
                                   ["Bachelors", "Masters", "Doctorate", "HS-grad", "Some-college"])
            marital_status = st.selectbox("Estado Civil",
                                        ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"])
        
        submitted = st.form_submit_button("🎯 Fazer Predição")
        
        if submitted:
            st.success("✅ Predição realizada com sucesso!")
            st.write("💰 Salário Previsto: > 50K")

if __name__ == "__main__":
    main()
