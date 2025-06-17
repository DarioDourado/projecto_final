"""
Dashboard Streamlit Completo - Análise Salarial Académica VERSÃO FINAL
Sistema interativo com todas as funcionalidades implementadas + Sistema de Login
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import logging
import warnings
from datetime import datetime, timedelta
import io
import hashlib
import json
import time

# Configurações
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="Análise Salarial - Dashboard Académico",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SISTEMA DE AUTENTICAÇÃO COMPLETO
# =============================================================================

class AuthenticationSystem:
    """Sistema de autenticação moderno para o dashboard"""
    
    def __init__(self):
        self.users_file = Path("config/users.json")
        self.sessions_file = Path("config/sessions.json")
        self.init_files()
        
        # Usuários padrão
        self.default_users = {
            "admin": {
                "password": self.hash_password("admin123"),
                "role": "admin",
                "name": "Administrador",
                "email": "admin@dashboard.com",
                "created": datetime.now().isoformat()
            },
            "demo": {
                "password": self.hash_password("demo123"),
                "role": "user", 
                "name": "Usuário Demo",
                "email": "demo@dashboard.com",
                "created": datetime.now().isoformat()
            },
            "guest": {
                "password": self.hash_password("guest123"),
                "role": "guest",
                "name": "Visitante",
                "email": "guest@dashboard.com", 
                "created": datetime.now().isoformat()
            }
        }
        
        self.ensure_default_users()
    
    def init_files(self):
        """Inicializar arquivos de configuração"""
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        if not self.users_file.exists():
            self.users_file.write_text("{}", encoding='utf-8')
        
        if not self.sessions_file.exists():
            self.sessions_file.write_text("{}", encoding='utf-8')
    
    def hash_password(self, password):
        """Hash seguro da senha"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def load_users(self):
        """Carregar usuários do arquivo"""
        try:
            return json.loads(self.users_file.read_text(encoding='utf-8'))
        except:
            return {}
    
    def save_users(self, users):
        """Salvar usuários no arquivo"""
        try:
            self.users_file.write_text(json.dumps(users, indent=2), encoding='utf-8')
        except Exception as e:
            st.error(f"Erro ao salvar usuários: {e}")
    
    def ensure_default_users(self):
        """Garantir que usuários padrão existam"""
        users = self.load_users()
        updated = False
        
        for username, user_data in self.default_users.items():
            if username not in users:
                users[username] = user_data
                updated = True
        
        if updated:
            self.save_users(users)
    
    def authenticate(self, username, password):
        """Autenticar usuário"""
        users = self.load_users()
        
        if username in users:
            stored_password = users[username]["password"]
            if stored_password == self.hash_password(password):
                return users[username]
        
        return None
    
    def create_session(self, username, user_data):
        """Criar sessão de usuário"""
        session_id = hashlib.md5(f"{username}{time.time()}".encode()).hexdigest()
        
        try:
            sessions = json.loads(self.sessions_file.read_text(encoding='utf-8'))
        except:
            sessions = {}
        
        sessions[session_id] = {
            "username": username,
            "user_data": user_data,
            "created": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
        
        # Limpar sessões antigas (>24h)
        cutoff = datetime.now() - timedelta(hours=24)
        sessions = {
            sid: sdata for sid, sdata in sessions.items()
            if datetime.fromisoformat(sdata["last_activity"]) > cutoff
        }
        
        try:
            self.sessions_file.write_text(json.dumps(sessions, indent=2), encoding='utf-8')
        except:
            pass
        
        return session_id
    
    def get_session(self, session_id):
        """Obter dados da sessão"""
        try:
            sessions = json.loads(self.sessions_file.read_text(encoding='utf-8'))
            if session_id in sessions:
                session = sessions[session_id]
                # Verificar se não expirou
                last_activity = datetime.fromisoformat(session["last_activity"])
                if datetime.now() - last_activity < timedelta(hours=24):
                    # Atualizar última atividade
                    session["last_activity"] = datetime.now().isoformat()
                    sessions[session_id] = session
                    self.sessions_file.write_text(json.dumps(sessions, indent=2), encoding='utf-8')
                    return session
        except:
            pass
        
        return None
    
    def logout(self, session_id):
        """Fazer logout"""
        try:
            sessions = json.loads(self.sessions_file.read_text(encoding='utf-8'))
            if session_id in sessions:
                del sessions[session_id]
                self.sessions_file.write_text(json.dumps(sessions, indent=2), encoding='utf-8')
        except:
            pass
    
    def register_user(self, username, password, name, email, role="user"):
        """Registrar novo usuário"""
        users = self.load_users()
        
        if username in users:
            return False, "Usuário já existe"
        
        users[username] = {
            "password": self.hash_password(password),
            "role": role,
            "name": name,
            "email": email,
            "created": datetime.now().isoformat()
        }
        
        self.save_users(users)
        return True, "Usuário criado com sucesso"

# Instância global do sistema de autenticação
auth_system = AuthenticationSystem()

def show_login_page():
    """Página de login modernizada"""
    
    # CSS customizado para login
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        color: white;
    }
    .login-title {
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .login-form {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    .demo-info {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        backdrop-filter: blur(5px);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header da aplicação
    st.markdown('<div class="login-title">💰 Dashboard Salarial</div>', unsafe_allow_html=True)
    
    # Container principal
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            
            # Tabs de login e registro
            tab1, tab2 = st.tabs(["🔐 Login", "📝 Registro"])
            
            with tab1:
                st.markdown('<div class="login-form">', unsafe_allow_html=True)
                
                with st.form("login_form"):
                    st.markdown("### 🔓 Acesso ao Sistema")
                    
                    username = st.text_input("👤 Usuário:", placeholder="Digite seu usuário")
                    password = st.text_input("🔑 Senha:", type="password", placeholder="Digite sua senha")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        login_button = st.form_submit_button("🚀 Entrar", use_container_width=True)
                    with col2:
                        demo_button = st.form_submit_button("🎮 Demo", use_container_width=True)
                    
                    if login_button and username and password:
                        user_data = auth_system.authenticate(username, password)
                        if user_data:
                            session_id = auth_system.create_session(username, user_data)
                            st.session_state.session_id = session_id
                            st.session_state.user_data = user_data
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.success(f"✅ Bem-vindo, {user_data['name']}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Credenciais inválidas!")
                    
                    if demo_button:
                        # Login automático como demo
                        user_data = auth_system.authenticate("demo", "demo123")
                        if user_data:
                            session_id = auth_system.create_session("demo", user_data)
                            st.session_state.session_id = session_id
                            st.session_state.user_data = user_data
                            st.session_state.authenticated = True
                            st.session_state.username = "demo"  # Salvar username
                            st.success("✅ Entrando como usuário demo...")
                            time.sleep(1)
                            st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown('<div class="login-form">', unsafe_allow_html=True)
                
                with st.form("register_form"):
                    st.markdown("### 📝 Criar Conta")
                    
                    new_username = st.text_input("👤 Novo Usuário:", placeholder="Escolha um usuário")
                    new_name = st.text_input("👨‍💼 Nome Completo:", placeholder="Seu nome completo")
                    new_email = st.text_input("📧 Email:", placeholder="seu@email.com")
                    new_password = st.text_input("🔑 Senha:", type="password", placeholder="Escolha uma senha")
                    confirm_password = st.text_input("🔑 Confirmar Senha:", type="password", placeholder="Confirme a senha")
                    
                    register_button = st.form_submit_button("✨ Criar Conta", use_container_width=True)
                    
                    if register_button:
                        if not all([new_username, new_name, new_email, new_password]):
                            st.error("❌ Preencha todos os campos!")
                        elif new_password != confirm_password:
                            st.error("❌ Senhas não coincidem!")
                        elif len(new_password) < 6:
                            st.error("❌ Senha deve ter pelo menos 6 caracteres!")
                        else:
                            success, message = auth_system.register_user(
                                new_username, new_password, new_name, new_email
                            )
                            if success:
                                st.success(f"✅ {message}")
                                st.info("🔄 Agora você pode fazer login!")
                            else:
                                st.error(f"❌ {message}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Informações de demo
            st.markdown("""
            <div class="demo-info">
                <h4>🎮 Contas de Demonstração:</h4>
                <ul>
                    <li><strong>admin</strong> / admin123 (Administrador)</li>
                    <li><strong>demo</strong> / demo123 (Usuário)</li>
                    <li><strong>guest</strong> / guest123 (Visitante)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

def check_authentication():
    """Verificar autenticação do usuário"""
    
    # Inicializar estado de autenticação
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    
    # Verificar sessão existente
    if st.session_state.session_id:
        session = auth_system.get_session(st.session_state.session_id)
        if session:
            st.session_state.authenticated = True
            st.session_state.user_data = session["user_data"]
            st.session_state.username = session["username"]
            return True
        else:
            # Sessão expirada
            st.session_state.authenticated = False
            st.session_state.session_id = None
            st.session_state.user_data = None
            st.session_state.username = None
    
    return st.session_state.authenticated

def show_user_info():
    """Mostrar informações do usuário na sidebar"""
    if 'user_data' in st.session_state and st.session_state.user_data:
        user = st.session_state.user_data
        username = st.session_state.get('username', 'N/A')
        
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 Usuário Logado")
            
            st.markdown(f"""
            <div class="user-info-card">
                <h4>🎭 {user['name']}</h4>
                <p><strong>👤 Usuário:</strong> {username}</p>
                <p><strong>🎯 Papel:</strong> {user['role'].title()}</p>
                <p><strong>📧 Email:</strong> {user['email']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚪 Logout", use_container_width=True):
                if st.session_state.session_id:
                    auth_system.logout(st.session_state.session_id)
                
                # Limpar sessão
                st.session_state.authenticated = False
                st.session_state.session_id = None
                st.session_state.user_data = None
                st.session_state.username = None
                st.session_state.current_page = "📊 Visão Geral"
                
                st.success("✅ Logout realizado com sucesso!")
                time.sleep(1)
                st.rerun()

# =============================================================================
# CSS MELHORADO
# =============================================================================

def apply_custom_css():
    """Aplicar CSS customizado melhorado"""
    st.markdown("""
<style>
    /* Importar fontes do Google */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Reset e configurações gerais */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header principal */
    .main-header {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Cards de métricas */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    
    .metric-card h2 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Boxes de status */
    .success-box {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        color: #2d3436;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .error-box {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Botões customizados */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Sidebar customizada */
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    
    /* Card de informações do usuário */
    .user-info-card {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .user-info-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
    }
    
    .user-info-card p {
        margin: 0.3rem 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Tabs customizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f8f9fa;
        border-radius: 10px;
        color: #495057;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Melhorias no DataFrame */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Loading spinner customizado */
    .stSpinner {
        text-align: center;
    }
    
    /* Métricas do Streamlit */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        padding: 1rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="metric-container"] > div {
        color: white;
    }
    
    /* Expander customizado */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Hide hamburger menu e footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsividade */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .metric-card h2 {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FUNÇÕES AUXILIARES OTIMIZADAS
# =============================================================================

@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_data():
    """Carregar dados com cache otimizado"""
    try:
        # Prioridade: dados processados
        data_paths = [
            Path("data/processed/data_processed.csv"),
            Path("data/raw/4-Carateristicas_salario.csv"),
            Path("bkp/4-Carateristicas_salario.csv"),
            Path("4-Carateristicas_salario.csv")
        ]
        
        for path in data_paths:
            if path.exists():
                df = pd.read_csv(path)
                df = clean_dataframe(df)
                status = "✅ Dados processados" if "processed" in str(path) else "⚠️ Dados brutos"
                return df, f"{status} carregados: {path.name}"
        
        return None, "❌ Nenhum arquivo de dados encontrado!"
    except Exception as e:
        return None, f"❌ Erro ao carregar dados: {e}"

def clean_dataframe(df):
    """Limpeza otimizada do DataFrame"""
    # Converter categóricas
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).replace(['None', 'nan', 'NaN', '?'], 'Unknown')
    
    # Garantir numéricas
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

@st.cache_data
def load_analysis_files():
    """Carregar análises com cache"""
    files_status = {
        'images': [],
        'analysis': [],
        'models': []
    }
    
    # Buscar em múltiplas localizações
    search_paths = {
        'images': [Path("output/images"), Path("imagens"), Path("bkp/imagens")],
        'analysis': [Path("output/analysis"), Path("output"), Path(".")],
        'models': [Path("data/processed"), Path("bkp"), Path(".")]
    }
    
    for category, paths in search_paths.items():
        for path in paths:
            if path.exists():
                if category == 'images':
                    files_status[category].extend(list(path.glob("*.png")))
                    files_status[category].extend(list(path.glob("*.jpg")))
                elif category == 'analysis':
                    files_status[category].extend(list(path.glob("*.csv")))
                    files_status[category].extend(list(path.glob("*.md")))
                elif category == 'models':
                    files_status[category].extend(list(path.glob("*.joblib")))
                    files_status[category].extend(list(path.glob("*.pkl")))
    
    # Remover duplicatas
    for category in files_status:
        files_status[category] = list(set(files_status[category]))
    
    return files_status

def create_comparison_chart(df, metric='accuracy'):
    """Criar gráfico de comparação otimizado"""
    if df.empty:
        return None
    
    fig = go.Figure()
    
    # Ordenar por métrica
    df_sorted = df.sort_values(metric, ascending=True)
    
    # Criar barras com gradiente
    colors = px.colors.qualitative.Set3[:len(df_sorted)]
    
    fig.add_trace(go.Bar(
        x=df_sorted[metric],
        y=df_sorted.index,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(50, 50, 50, 0.5)', width=1)
        ),
        text=[f'{x:.3f}' for x in df_sorted[metric]],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f'Comparação de Modelos - {metric.title()}',
        xaxis_title=metric.title(),
        yaxis_title='Modelos',
        height=400,
        showlegend=False
    )
    
    return fig

# =============================================================================
# FUNÇÕES DE GRÁFICOS MODERNIZADOS
# =============================================================================

def create_modern_pie_chart(data, values, names, title, color_scheme='viridis'):
    """Criar gráfico de pizza moderno com efeitos - CORRIGIDO"""
    
    # Verificar se values é array numpy e converter se necessário
    if hasattr(values, 'values'):
        values_list = values.values.tolist()
    elif hasattr(values, 'tolist'):
        values_list = values.tolist()
    else:
        values_list = list(values)
    
    # Verificar se names é array e converter
    if hasattr(names, 'values'):
        names_list = names.values.tolist()
    elif hasattr(names, 'tolist'):
        names_list = names.tolist()
    else:
        names_list = list(names)
    
    # Encontrar o índice do valor máximo de forma segura
    max_index = values_list.index(max(values_list))
    
    fig = go.Figure(data=[go.Pie(
        labels=names_list,
        values=values_list,
        hole=0.4,  # Donut style
        textinfo='label+percent+value',
        textfont_size=12,
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='#FFFFFF', width=2)
        ),
        pull=[0.05 if i == max_index else 0 for i in range(len(values_list))],  # Destacar maior fatia
        hovertemplate='<b>%{label}</b><br>Quantidade: %{value}<br>Percentual: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(size=18, color='#2c3e50')
        ),
        font=dict(family="Arial, sans-serif", size=12),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        ),
        margin=dict(t=60, b=40, l=40, r=120),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_modern_bar_chart(data, x, y, title, color_column=None, orientation='v'):
    """Criar gráfico de barras moderno com gradientes - CORRIGIDO"""
    if orientation == 'v':
        fig = px.bar(
            data, x=x, y=y, 
            color=color_column if color_column else y,
            color_continuous_scale='viridis',
            title=title
        )
        fig.update_traces(
            marker_line_color='rgba(255,255,255,0.8)',
            marker_line_width=2,
            texttemplate='%{y}',
            textposition='outside'
        )
    else:
        fig = px.bar(
            data, x=y, y=x, 
            color=color_column if color_column else y,
            color_continuous_scale='plasma',
            orientation='h',
            title=title
        )
        fig.update_traces(
            marker_line_color='rgba(255,255,255,0.8)',
            marker_line_width=2,
            texttemplate='%{x}',
            textposition='outside'
        )
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(size=18, color='#2c3e50')
        ),
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif"),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        )
    )
    
    return fig

def create_modern_scatter_plot(data, x, y, color=None, size=None, title="", hover_data=None):
    """Criar scatter plot moderno com animações - CORRIGIDO"""
    
    # Verificar se a coluna size existe e tem valores válidos
    size_col = None
    if size and size in data.columns:
        # Verificar se há valores não-nulos e variação
        size_values = data[size].dropna()
        if len(size_values) > 0 and size_values.var() > 0:
            size_col = size
    
    fig = px.scatter(
        data, x=x, y=y,
        color=color,
        size=size_col,
        hover_data=hover_data,
        color_continuous_scale='viridis',
        title=title,
        opacity=0.7
    )
    
    # Atualizar traces com verificação
    marker_dict = {
        'line': dict(width=1, color='rgba(255,255,255,0.8)')
    }
    
    if size_col:
        max_size = data[size_col].max()
        if max_size > 0:
            marker_dict.update({
                'sizemode': 'diameter',
                'sizeref': 2. * max_size / (40.**2)
            })
    
    fig.update_traces(marker=marker_dict)
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(size=18, color='#2c3e50')
        ),
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif"),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        )
    )
    
    return fig

def create_modern_histogram(data, column, title, bins=30, color_column=None):
    """Criar histograma moderno com densidade - CORRIGIDO"""
    
    # Verificar se a coluna existe
    if column not in data.columns:
        st.error(f"Coluna '{column}' não encontrada no dataset")
        return None
    
    # Verificar se há dados válidos
    valid_data = data[column].dropna()
    if len(valid_data) == 0:
        st.warning(f"Nenhum dado válido encontrado na coluna '{column}'")
        return None
    
    try:
        fig = px.histogram(
            data, x=column,
            nbins=bins,
            color=color_column,
            marginal="box",  # Adicionar boxplot
            title=title,
            opacity=0.7
        )
        
        fig.update_traces(
            marker_line_color='rgba(255,255,255,0.8)',
            marker_line_width=2
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(size=18, color='#2c3e50')
            ),
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif"),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erro ao criar histograma: {e}")
        return None

def create_interactive_correlation_heatmap(df):
    """Criar heatmap de correlação interativo - CORRIGIDO"""
    
    # Selecionar apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        st.warning("Poucas variáveis numéricas para análise de correlação")
        return None
    
    try:
        # Calcular matriz de correlação
        corr_matrix = df[numeric_cols].corr()
        
        # Criar máscara para triângulo superior
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        masked_corr = corr_matrix.where(~mask)
        
        # Preparar dados para o heatmap
        z_data = masked_corr.values
        x_labels = masked_corr.columns.tolist()
        y_labels = masked_corr.index.tolist()
        
        # Criar texto para anotações
        text_data = np.round(z_data, 2).astype(str)
        text_data[mask] = ''  # Remover texto da parte mascarada
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale='RdBu',
            zmid=0,
            text=text_data,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlação: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='<b>🔗 Matriz de Correlação Interativa</b>',
                x=0.5,
                font=dict(size=18, color='#2c3e50')
            ),
            xaxis_title='<b>Variáveis</b>',
            yaxis_title='<b>Variáveis</b>',
            font=dict(family="Arial, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erro ao criar heatmap de correlação: {e}")
        return None

def apply_filters(df, filters):
    """Aplicar filtros de forma otimizada"""
    filtered_df = df.copy()
    
    for col, values in filters.items():
        if col in df.columns:
            if isinstance(values, list) and values:
                filtered_df = filtered_df[filtered_df[col].isin(values)]
            elif isinstance(values, tuple) and len(values) == 2:
                filtered_df = filtered_df[
                    (filtered_df[col] >= values[0]) & 
                    (filtered_df[col] <= values[1])
                ]
    
    return filtered_df

def show_admin_config():
    """Página de configurações administrativas"""
    st.header("⚙️ Configurações Administrativas")
    
    if st.session_state.user_data.get('role') != 'admin':
        st.error("❌ Acesso restrito a administradores!")
        return
    
    tab1, tab2, tab3 = st.tabs(["👥 Usuários", "📊 Sistema", "🔧 Manutenção"])
    
    with tab1:
        st.subheader("👥 Gerenciamento de Usuários")
        
        # Listar usuários
        users = auth_system.load_users()
        
        if users:
            users_df = pd.DataFrame([
                {
                    'Usuário': username,
                    'Nome': user_data['name'],
                    'Email': user_data['email'],
                    'Papel': user_data['role'],
                    'Criado': user_data.get('created', 'N/A')
                }
                for username, user_data in users.items()
            ])
            
            st.dataframe(users_df, use_container_width=True)
            
            # Remover usuário
            st.subheader("🗑️ Remover Usuário")
            user_to_remove = st.selectbox("Selecionar usuário:", list(users.keys()))
            
            if st.button("❌ Remover Usuário") and user_to_remove:
                if user_to_remove == 'admin':
                    st.error("❌ Não é possível remover o usuário admin!")
                else:
                    del users[user_to_remove]
                    auth_system.save_users(users)
                    st.success(f"✅ Usuário '{user_to_remove}' removido!")
                    st.rerun()
        else:
            st.info("Nenhum usuário encontrado.")
    
    with tab2:
        st.subheader("📊 Informações do Sistema")
        
        # Estatísticas do sistema
        users = auth_system.load_users()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("👥 Total de Usuários", len(users))
        
        with col2:
            admin_count = sum(1 for u in users.values() if u.get('role') == 'admin')
            st.metric("🛡️ Administradores", admin_count)
        
        with col3:
            # Contagem de sessões ativas
            try:
                sessions = json.loads(auth_system.sessions_file.read_text(encoding='utf-8'))
                active_sessions = len(sessions)
            except:
                active_sessions = 0
            st.metric("🔗 Sessões Ativas", active_sessions)
        
        # Informações de arquivos
        st.subheader("📁 Status dos Arquivos")
        files_status = load_analysis_files()
        
        for category, files in files_status.items():
            st.write(f"**{category.title()}:** {len(files)} arquivos")
    
    with tab3:
        st.subheader("🔧 Manutenção do Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Limpar Sessões Expiradas"):
                try:
                    sessions = json.loads(auth_system.sessions_file.read_text(encoding='utf-8'))
                    cutoff = datetime.now() - timedelta(hours=24)
                    active_sessions = {
                        sid: sdata for sid, sdata in sessions.items()
                        if datetime.fromisoformat(sdata["last_activity"]) > cutoff
                    }
                    auth_system.sessions_file.write_text(json.dumps(active_sessions, indent=2), encoding='utf-8')
                    st.success("✅ Sessões expiradas removidas!")
                except Exception as e:
                    st.error(f"❌ Erro: {e}")
        
        with col2:
            if st.button("🔄 Recarregar Cache"):
                # Limpar caches do Streamlit
                load_data.clear()
                load_analysis_files.clear()
                st.success("✅ Cache recarregado!")
                st.rerun()

try:
    from src.pages.prediction import show_prediction_page
except ImportError:
    def show_prediction_page(data):
        st.error("❌ Módulo de predição não encontrado")
        st.info("Verifique se o arquivo src/pages/prediction.py existe")

# ADICIONAR ESTES IMPORTS:
try:
    from src.pages.overview import show_overview_page
    from src.pages.exploratory import show_exploratory_page  
    from src.pages.models import show_models_page
    from src.pages.admin import show_admin_page
    from src.pages.prediction import show_prediction_page
    from src.pages.clustering import show_clustering_page
    from src.pages.association_rules import show_association_rules_page
    PAGES_IMPORTED = True
except ImportError as e:
    st.error(f"❌ Erro ao importar páginas: {e}")
    PAGES_IMPORTED = False

def main():
    """Interface principal com autenticação"""
    
    # Aplicar CSS customizado
    apply_custom_css()
    
    # Verificar autenticação
    if not check_authentication():
        show_login_page()
        return
    
    # Mostrar informações do usuário
    show_user_info()
    
    # Header elegante
    st.markdown('<div class="main-header">💰 Dashboard de Análise Salarial</div>', 
                unsafe_allow_html=True)
    
    # Verificar permissões por papel
    user_role = st.session_state.user_data.get('role', 'guest')
    
    # Carregar dados
    df, load_message = load_data()
    files_status = load_analysis_files()
    
    # Sidebar melhorada
    with st.sidebar:
        st.markdown("## 🎛️ Controle Central")
        
        # Status do sistema com cores
        pipeline_executed = len(files_status['models']) > 0
        if pipeline_executed:
            st.markdown('<div class="success-box">✅ Pipeline Executado!</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ Execute: python main.py</div>', 
                       unsafe_allow_html=True)
        
        # Métricas do sistema
        st.markdown("### 📊 Arquivos Encontrados")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎨 Imagens", len(files_status['images']))
            st.metric("📊 Análises", len(files_status['analysis']))
        with col2:
            st.metric("🤖 Modelos", len(files_status['models']))
            if df is not None:
                st.metric("📋 Registros", f"{len(df):,}")
        
        # Navegação baseada em permissões
        st.markdown("### 🧭 Navegação")
        
        # Páginas disponíveis por papel
        all_pages = [
            ("📊 Visão Geral", "overview", ["admin", "user", "guest"]),
            ("📈 Análise Exploratória", "exploratory", ["admin", "user", "guest"]),
            ("🤖 Modelos ML", "models", ["admin", "user"]),
            ("🎯 Clustering", "clustering", ["admin", "user"]),
            ("📋 Regras de Associação", "rules", ["admin", "user"]),
            ("📊 Métricas Avançadas", "metrics", ["admin", "user"]),
            ("🔮 Predição", "prediction", ["admin", "user"]),
            ("📁 Relatórios", "reports", ["admin", "user", "guest"]),
            ("⚙️ Configurações", "config", ["admin"])
        ]
        
        # Filtrar páginas por papel
        available_pages = [(name, key) for name, key, roles in all_pages if user_role in roles]
        
        for page_name, page_key in available_pages:
            if st.button(page_name, key=f"nav_{page_key}"):
                st.session_state.current_page = page_name
        
        # Filtros se dados disponíveis e permitidos
        if df is not None and len(df) > 0 and user_role in ['admin', 'user']:
            st.markdown("### 🔍 Filtros")
            
            # Reset filters button
            if st.button("🗑️ Limpar Filtros"):
                st.session_state.filters = {}
                st.rerun()
            
            # Salary filter
            if 'salary' in df.columns:
                salary_options = [str(x) for x in df['salary'].unique() if str(x) != 'Unknown']
                if salary_options:
                    salary_filter = st.multiselect("💰 Salário", salary_options)
                    if salary_filter:
                        st.session_state.filters['salary'] = salary_filter
            
            # Age filter
            if 'age' in df.columns:
                age_range = st.slider("🎂 Idade", 
                                    int(df['age'].min()), 
                                    int(df['age'].max()),
                                    (int(df['age'].min()), int(df['age'].max())))
                if age_range != (int(df['age'].min()), int(df['age'].max())):
                    st.session_state.filters['age'] = age_range
        
        # Status de filtros
        if hasattr(st.session_state, 'filters') and st.session_state.filters:
            st.markdown("### 📋 Filtros Ativos")
            for filter_name, filter_value in st.session_state.filters.items():
                if isinstance(filter_value, list):
                    st.write(f"• **{filter_name}**: {', '.join(map(str, filter_value))}")
                elif isinstance(filter_value, tuple):
                    st.write(f"• **{filter_name}**: {filter_value[0]} - {filter_value[1]}")
    
    # Inicializar filtros se não existirem
    if 'filters' not in st.session_state:
        st.session_state.filters = {}
    
    # Aplicar filtros
    if df is not None:
        filtered_df = apply_filters(df, st.session_state.filters)
        if len(filtered_df) != len(df):
            st.info(f"🔍 Filtros: {len(filtered_df):,} de {len(df):,} registros ({len(filtered_df)/len(df):.1%})")
    else:
        filtered_df = None
    
    # Verificação de dados
    if filtered_df is None or len(filtered_df) == 0:
        st.error("❌ Nenhum dado disponível após filtros")
        st.info("Verifique os filtros ou execute: `python main.py`")
        return
    
    # Inicializar página atual se não existir
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📊 Visão Geral"
    
    # Roteamento de páginas
    current_page = st.session_state.current_page
    
    # Verificar se usuário tem acesso à página
    page_access = {
        "📊 Visão Geral": ["admin", "user", "guest"],
        "📈 Análise Exploratória": ["admin", "user", "guest"],
        "🤖 Modelos ML": ["admin", "user"],
        "🎯 Clustering": ["admin", "user"],
        "📋 Regras de Associação": ["admin", "user"],
        "📊 Métricas Avançadas": ["admin", "user"],
        "🔮 Predição": ["admin", "user"],
        "📁 Relatórios": ["admin", "user", "guest"],
        "⚙️ Configurações": ["admin"]
    }
    
    if current_page in page_access and user_role not in page_access[current_page]:
        st.error(f"❌ Acesso negado! Papel '{user_role}' não tem permissão para '{current_page}'")
        st.session_state.current_page = "📊 Visão Geral"
        st.rerun()
        return
    
    # Executar página correspondente
    try:
        if current_page == "📊 Visão Geral":
            show_overview_enhanced(filtered_df, load_message, files_status)
        elif current_page == "📈 Análise Exploratória":
            show_exploratory_analysis_enhanced(filtered_df)
        elif current_page == "🤖 Modelos ML":
            show_ml_models_enhanced(filtered_df, files_status)
        elif current_page == "🎯 Clustering":
            show_clustering_analysis_enhanced(filtered_df, files_status)
        elif current_page == "📋 Regras de Associação":
            show_association_rules_enhanced(filtered_df, files_status)
        elif current_page == "📊 Métricas Avançadas":
            show_advanced_metrics_enhanced(filtered_df, files_status)
        elif current_page == "🔮 Predição":
            # ALTERAR ESTA LINHA:
            # show_prediction_interface_enhanced(filtered_df, files_status)  # Função antiga
            # PARA:
            data_dict = {'original': filtered_df}  # Preparar dados no formato esperado
            show_prediction_page(data_dict)  # Nova função
        elif current_page == "📁 Relatórios":
            show_reports_enhanced(files_status)
        elif current_page == "⚙️ Configurações":
            show_admin_config()
        else:
            st.error("❌ Página não encontrada!")
            
    except Exception as e:
        st.error(f"❌ Erro na página '{current_page}': {e}")
        st.info("Tente navegar para outra página ou recarregue a aplicação.")

# =============================================================================
# FUNÇÕES DE PÁGINAS IMPLEMENTADAS
# =============================================================================

def show_overview_enhanced(df, load_message, files_status):
    """Visão geral com gráficos modernizados"""
    st.header("📊 Visão Geral do Dataset")
    
    # Status message com estilo
    if "processados" in load_message:
        st.success(load_message)
    else:
        st.warning(load_message)
    
    # Métricas principais com cards modernos
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 Registros</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Colunas</h3>
            <h2>{len(df.columns)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'salary' in df.columns:
            high_salary_rate = (df['salary'] == '>50K').mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Salário +50k</h3>
                <h2>{high_salary_rate:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Salário +50k</h3>
                <h2>N/A</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
        st.markdown(f"""
        <div class="metric-card">
            <h3>❌ Dados Nulos</h3>
            <h2>{missing_rate:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráficos principais modernizados
    st.subheader("📈 Distribuições Principais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'salary' in df.columns:
            salary_counts = df['salary'].value_counts()
            fig = create_modern_pie_chart(
                data=None,
                values=salary_counts.values,
                names=salary_counts.index,
                title="💰 Distribuição de Salário"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Coluna 'salary' não encontrada")
    
    with col2:
        if 'sex' in df.columns:
            sex_data = df['sex'].value_counts().reset_index()
            sex_data.columns = ['sex', 'count']
            fig = create_modern_bar_chart(
                data=sex_data,
                x='sex',
                y='count',
                title="👥 Distribuição por Sexo"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Coluna 'sex' não encontrada")
    
    # Gráficos adicionais com verificação de erro
    st.subheader("📈 Análises Detalhadas")
    
    tab1, tab2, tab3 = st.tabs(["📊 Distribuições", "🔗 Correlações", "📋 Estatísticas"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'age' in df.columns:
                fig = create_modern_histogram(
                    data=df,
                    column='age',
                    title="📊 Distribuição de Idade",
                    color_column='salary' if 'salary' in df.columns else None
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Coluna 'age' não encontrada")
        
        with col2:
            if 'education-num' in df.columns:
                fig = create_modern_histogram(
                    data=df,
                    column='education-num',
                    title="🎓 Anos de Educação",
                    color_column='salary' if 'salary' in df.columns else None
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Coluna 'education-num' não encontrada")
    
    with tab2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            fig = create_interactive_correlation_heatmap(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Poucas variáveis numéricas para análise de correlação")
    
    with tab3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
        else:
            st.info("Nenhuma variável numérica encontrada")

def show_exploratory_analysis_enhanced(df):
    """Análise exploratória com gráficos modernizados"""
    st.header("📈 Análise Exploratória Avançada")
    
    # Controles modernizados
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                x_var = st.selectbox("🔢 Variável X:", numeric_cols)
        
        with col2:
            y_var = st.selectbox("📊 Variável Y:", ["Nenhuma"] + numeric_cols)
        
        with col3:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                color_var = st.selectbox("🎨 Cor por:", ["Nenhuma"] + categorical_cols)
            else:
                color_var = "Nenhuma"
    
    # Gráficos baseados na seleção
    if 'x_var' in locals():
        if y_var != "Nenhuma":
            # Scatter plot
            fig = create_modern_scatter_plot(
                data=df,
                x=x_var,
                y=y_var,
                color=color_var if color_var != "Nenhuma" else None,
                title=f"📊 {x_var} vs {y_var}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Histograma
            fig = create_modern_histogram(
                data=df,
                column=x_var,
                title=f"📊 Distribuição de {x_var}",
                color_column=color_var if color_var != "Nenhuma" else None
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # Análises por categoria
    st.subheader("🎯 Análises por Categoria")
    
    if 'salary' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'workclass' in df.columns:
                workclass_salary = df.groupby('workclass')['salary'].apply(
                    lambda x: (x == '>50K').mean()
                ).reset_index()
                workclass_salary.columns = ['workclass', 'high_salary_rate']
                
                fig = create_modern_bar_chart(
                    data=workclass_salary,
                    x='workclass',
                    y='high_salary_rate',
                    title="💼 Taxa de Salário Alto por Classe Trabalhadora"
                )
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'marital-status' in df.columns:
                marital_salary = df.groupby('marital-status')['salary'].apply(
                    lambda x: (x == '>50K').mean()
                ).reset_index()
                marital_salary.columns = ['marital_status', 'high_salary_rate']
                
                fig = create_modern_bar_chart(
                    data=marital_salary,
                    x='marital_status',
                    y='high_salary_rate',
                    title="💑 Taxa de Salário Alto por Estado Civil"
                )
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Coluna 'marital-status' não encontrada")

def show_ml_models_enhanced(df, files_status):
    """Modelos ML implementados"""
    st.header("🤖 Modelos de Machine Learning")
    
    # Verificar se há modelos
    if not files_status['models']:
        st.warning("⚠️ Nenhum modelo encontrado. Execute: python main.py")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Comparação", "🎯 Features", "📈 Performance"])
    
    with tab1:
        # Procurar arquivo de comparação
        comparison_file = None
        for file in files_status['analysis']:
            if 'model_comparison' in file.name or 'comparison' in file.name:
                comparison_file = file
                break
        
        if comparison_file:
            try:
                comparison_df = pd.read_csv(comparison_file)
                if 'model_name' in comparison_df.columns:
                    comparison_df.set_index('model_name', inplace=True)
                
                st.dataframe(comparison_df.round(4), use_container_width=True)
                
                # Gráfico de comparação
                if 'accuracy' in comparison_df.columns:
                    fig = create_comparison_chart(comparison_df, 'accuracy')
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erro ao carregar comparação: {e}")
        else:
            st.info("Execute o pipeline para gerar comparação de modelos")
    
    with tab2:
        # Feature importance
        importance_files = [f for f in files_status['analysis'] 
                          if 'importance' in f.name or 'feature' in f.name]
        
        if importance_files:
            selected_file = st.selectbox("Modelo:", 
                                       [f.stem for f in importance_files])
            
            # Carregar arquivo selecionado
            for file in importance_files:
                if selected_file in file.stem:
                    try:
                        importance_df = pd.read_csv(file)
                        if 'feature' in importance_df.columns and 'importance' in importance_df.columns:
                            top_features = importance_df.head(15)
                            
                            fig = px.bar(top_features, x='importance', y='feature',
                                       orientation='h', title="Feature Importance")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.dataframe(importance_df.head(20), use_container_width=True)
                    except Exception as e:
                        st.error(f"Erro: {e}")
                    break
        else:
            st.info("Execute o pipeline para gerar feature importance")
    
    with tab3:
        # Imagens de performance
        performance_images = [f for f in files_status['images'] 
                            if any(keyword in f.name.lower() 
                                  for keyword in ['roc', 'curve', 'performance'])]
        
        if performance_images:
            for img in performance_images:
                st.image(str(img), caption=img.name, use_column_width=True)
        else:
            st.info("Execute o pipeline para gerar gráficos de performance")

def show_clustering_analysis_enhanced(df, files_status):
    """Análise de clustering"""
    st.header("🎯 Análise de Clustering Modernizada")
    
    # Verificar arquivos existentes
    clustering_files = {
        'analysis_image': Path("output/images/clustering_analysis.png"),
        'pca_image': Path("output/images/clusters_pca_visualization.png"),
        'cluster_data': Path("output/analysis/clustering_results.csv")
    }
    
    files_found = {name: path.exists() for name, path in clustering_files.items()}
    
    # Dashboard de status
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "✅" if files_found['analysis_image'] else "❌"
        st.markdown(f"""
        <div class="{'success-box' if files_found['analysis_image'] else 'warning-box'}">
            <h4>{status} Análise do Cotovelo</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = "✅" if files_found['pca_image'] else "❌"
        st.markdown(f"""
        <div class="{'success-box' if files_found['pca_image'] else 'warning-box'}">
            <h4>{status} Visualização PCA</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = "✅" if files_found['cluster_data'] else "❌"
        st.markdown(f"""
        <div class="{'success-box' if files_found['cluster_data'] else 'warning-box'}">
            <h4>{status} Dados de Cluster</h4>
        </div>
        """, unsafe_allow_html=True)
    
    # Clustering interativo modernizado
    st.subheader("🔧 Clustering Interativo Avançado")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            algorithm = st.selectbox("🤖 Algoritmo", ["K-Means", "DBSCAN", "Agglomerative"])
        
        with col2:
            n_clusters = st.slider("📊 Número de Clusters", 2, 10, 4)
        
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'salary' in numeric_cols:
                numeric_cols.remove('salary')
            
            features = st.multiselect(
                "🎯 Features",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
    
    if len(features) >= 2 and st.button("🚀 Executar Clustering Avançado", type="primary"):
        try:
            from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
            
            with st.spinner("🔄 Processando clustering..."):
                # Preparar dados
                X = df[features].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Aplicar algoritmo selecionado
                if algorithm == "K-Means":
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                elif algorithm == "DBSCAN":
                    clusterer = DBSCAN(eps=0.5, min_samples=5)
                else:  # Agglomerative
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                
                clusters = clusterer.fit_predict(X_scaled)
                
                # Calcular métricas
                if len(set(clusters)) > 1:
                    silhouette = silhouette_score(X_scaled, clusters)
                else:
                    silhouette = 0
                
                # Visualizações modernas
                col1, col2 = st.columns(2)
                
                with col1:
                    # Scatter plot 2D
                    if len(features) >= 2:
                        plot_df = X.copy()
                        plot_df['Cluster'] = [f'Cluster {i}' for i in clusters]
                        
                        fig = create_modern_scatter_plot(
                            data=plot_df,
                            x=features[0],
                            y=features[1],
                            color='Cluster',
                            title=f"🎯 Clustering {algorithm}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # PCA Visualization
                    if len(features) > 2:
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        pca_df = pd.DataFrame({
                            'PC1': X_pca[:, 0],
                            'PC2': X_pca[:, 1],
                            'Cluster': [f'Cluster {i}' for i in clusters]
                        })
                        
                        fig = create_modern_scatter_plot(
                            data=pca_df,
                            x='PC1',
                            y='PC2',
                            color='Cluster',
                            title="🎨 Visualização PCA"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Estatísticas dos clusters
                unique_clusters, counts = np.unique(clusters, return_counts=True)
                cluster_stats = pd.DataFrame({
                    'Cluster': [f'Cluster {i}' for i in unique_clusters],
                    'Tamanho': counts,
                    'Percentual': (counts / len(clusters) * 100).round(1)
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Estatísticas dos Clusters")
                    st.dataframe(cluster_stats, use_container_width=True)
                
                with col2:
                    # Gráfico de pizza dos clusters
                    fig = create_modern_pie_chart(
                        data=None,
                        values=counts,
                        names=[f'Cluster {i}' for i in unique_clusters],
                        title="📊 Distribuição dos Clusters"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Métricas de qualidade
                st.markdown("### 📈 Métricas de Qualidade")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Silhouette Score", f"{silhouette:.3f}")
                
                with col2:
                    st.metric("📊 Número de Clusters", len(unique_clusters))
                
                with col3:
                    st.metric("📋 Pontos Processados", len(X))
                
                st.success(f"✅ Clustering {algorithm} executado com sucesso!")
        
        except Exception as e:
            st.error(f"❌ Erro no clustering: {e}")
    
    # Mostrar imagens existentes se disponíveis
    if any(files_found.values()):
        st.subheader("📊 Análises Pré-computadas")
        
        for name, path in clustering_files.items():
            if path.exists() and path.suffix == '.png':
                st.image(str(path), caption=path.name, use_column_width=True)

def show_association_rules_enhanced(df, files_status):
    """Regras de associação"""
    st.header("📋 Regras de Associação")
    
    # Procurar arquivo de regras
    rules_files = [f for f in files_status['analysis'] 
                  if 'rule' in f.name.lower() or 'association' in f.name.lower()]
    
    if rules_files:
        for file in rules_files:
            try:
                rules_df = pd.read_csv(file)
                st.dataframe(rules_df, use_container_width=True)
                
                # Gráfico de suporte vs confiança
                if 'support' in rules_df.columns and 'confidence' in rules_df.columns:
                    fig = px.scatter(rules_df, x='support', y='confidence',
                                   hover_data=['antecedents', 'consequents'] if 'antecedents' in rules_df.columns else None,
                                   title="Suporte vs Confiança das Regras")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erro ao carregar regras: {e}")
    else:
        st.info("Execute o pipeline para gerar regras de associação")

def show_advanced_metrics_enhanced(df, files_status):
    """Métricas avançadas"""
    st.header("📊 Métricas Avançadas")
    
    # Procurar relatórios de métricas
    metrics_files = [f for f in files_status['analysis'] 
                    if 'metric' in f.name.lower() or 'advanced' in f.name.lower()]
    
    if metrics_files:
        for file in metrics_files:
            try:
                metrics_df = pd.read_csv(file)
                st.dataframe(metrics_df, use_container_width=True)
            except Exception as e:
                st.error(f"Erro: {e}")
    else:
        st.info("Execute o pipeline para gerar métricas avançadas")

def show_prediction_interface_enhanced(df, files_status):
    """Interface de predição"""
    st.header("🔮 Predição Interativa")
    
    # Verificar se há modelos
    if not files_status['models']:
        st.warning("⚠️ Nenhum modelo encontrado para predição")
        return
    
    # Tentar carregar um modelo
    model = None
    for model_file in files_status['models']:
        try:
            model = joblib.load(model_file)
            st.success(f"✅ Modelo carregado: {model_file.name}")
            break
        except Exception:
            continue
    
    if model is None:
        st.error("❌ Não foi possível carregar nenhum modelo")
        return
    
    st.info("Interface de predição em desenvolvimento. Valores configurados mas modelo precisa ser adaptado.")

def show_reports_enhanced(files_status):
    """Relatórios implementados"""
    st.header("📁 Relatórios Gerados")
    
    # Mostrar arquivos de relatório
    report_files = [f for f in files_status['analysis'] 
                   if f.suffix in ['.md', '.txt', '.csv']]
    
    if report_files:
        for file in report_files:
            with st.expander(f"📄 {file.name}"):
                try:
                    if file.suffix == '.md':
                        content = file.read_text(encoding='utf-8')
                        st.markdown(content)
                    elif file.suffix == '.csv':
                        df = pd.read_csv(file)
                        st.dataframe(df, use_container_width=True)
                    else:
                        content = file.read_text(encoding='utf-8')
                        st.text(content)
                except Exception as e:
                    st.error(f"Erro ao ler arquivo: {e}")
    else:
        st.info("Execute o pipeline para gerar relatórios")

if __name__ == "__main__":
    main()