"""
🎓 Dashboard Acadêmico Simplificado - Análise Salarial Científica
Versão estável e funcional
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import sys

# Adicionar src ao path para importações
sys.path.append(str(Path(__file__).parent / "src"))

# Importar as páginas
try:
    from src.pages.overview import show_overview_page
except ImportError:
    def show_overview_page(data):
        st.title("📊 Visão Geral")
        st.warning("⚠️ Módulo overview não encontrado")

try:
    from src.pages.clustering import show_clustering_page
except ImportError:
    def show_clustering_page(data):
        st.title("🎯 Clustering DBSCAN")
        st.warning("⚠️ Módulo clustering não encontrado")

try:
    from src.pages.association_rules import show_association_rules_page
except ImportError:
    def show_association_rules_page(data):
        st.title("🔗 Regras de Associação")
        st.warning("⚠️ Módulo association_rules não encontrado")

try:
    from src.pages.prediction import show_prediction_page
except ImportError:
    def show_prediction_page(data):
        st.title("🔮 Predição Salarial")
        st.warning("⚠️ Módulo prediction não encontrado")
        st.info("Sistema de predição não disponível")

# Configuração básica
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="🎓 Dashboard Acadêmico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SISTEMA DE LOGIN SIMPLIFICADO
# =============================================================================

def init_session_state():
    """Inicializar estado da sessão"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'username' not in st.session_state:
        st.session_state.username = None

def show_login():
    """Mostrar tela de login simplificada"""
    st.title("🎓 Dashboard Acadêmico")
    st.subheader("Sistema de Análise Salarial Científica")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        
        # Botões de acesso rápido
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("👨‍💼 Admin", use_container_width=True):
                st.session_state.logged_in = True
                st.session_state.user_role = "Administrator"
                st.session_state.username = "admin"
                st.rerun()
        
        with col_b:
            if st.button("👤 Demo", use_container_width=True):
                st.session_state.logged_in = True
                st.session_state.user_role = "Demo User"
                st.session_state.username = "demo"
                st.rerun()
        
        # Login manual
        with st.expander("🔑 Login Manual"):
            username = st.text_input("Usuário:")
            password = st.text_input("Senha:", type="password")
            
            if st.button("Entrar"):
                users = {
                    "admin": "admin123",
                    "demo": "demo123",
                    "guest": "guest123"
                }
                
                if username in users and users[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.user_role = username.capitalize()
                    st.session_state.username = username
                    st.success("Login realizado!")
                    st.rerun()
                else:
                    st.error("Credenciais inválidas!")
        
        # Info
        st.info("""
        **Credenciais de demonstração:**
        - admin / admin123
        - demo / demo123
        - guest / guest123
        """)

# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================

@st.cache_data
def load_data():
    """Carregar dados de análise"""
    try:
        data = {}
        analysis_dir = Path("output/analysis")
        
        if not analysis_dir.exists():
            return {}
        
        # Carregar CSVs
        csv_files = {
            'dbscan_results': 'dbscan_results.csv',
            'apriori_rules': 'apriori_rules.csv',
            'fp_growth_rules': 'fp_growth_rules.csv',
            'eclat_rules': 'eclat_rules.csv'
        }
        
        for key, filename in csv_files.items():
            file_path = analysis_dir / filename
            if file_path.exists():
                try:
                    data[key] = pd.read_csv(file_path)
                except:
                    continue
        
        # Carregar dados originais
        for source in ["bkp/4-Carateristicas_salario.csv"]:
            if Path(source).exists():
                try:
                    data['original'] = pd.read_csv(source)
                    break
                except:
                    continue
        
        return data
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return {}

# =============================================================================
# COMPONENTES VISUAIS BÁSICOS
# =============================================================================

def create_metric_card(title, value, description=""):
    """Criar card de métrica simples"""
    st.markdown(f"""
    <div style="
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <h4 style="margin: 0; color: #333;">{title}</h4>
        <h2 style="margin: 0.5rem 0; color: #667eea;">{value}</h2>
        <p style="margin: 0; color: #666; font-size: 0.9rem;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

def show_algorithm_status(data):
    """Mostrar status dos algoritmos"""
    st.subheader("🔄 Status dos Algoritmos")
    
    algorithms = {
        "DBSCAN": "dbscan_results",
        "APRIORI": "apriori_rules", 
        "FP-GROWTH": "fp_growth_rules",
        "ECLAT": "eclat_rules"
    }
    
    cols = st.columns(4)
    
    for i, (name, key) in enumerate(algorithms.items()):
        with cols[i]:
            if key in data and len(data[key]) > 0:
                count = len(data[key])
                st.success(f"✅ {name}")
                st.write(f"Resultados: {count}")
            else:
                st.error(f"❌ {name}")
                st.write("Não executado")

# =============================================================================
# PÁGINAS SIMPLIFICADAS
# =============================================================================

def show_overview_page(data):
    """Página de visão geral"""
    st.title("📊 Visão Geral")
    st.write("Dashboard de Análise Salarial Científica")
    
    if not data:
        st.warning("⚠️ Dados não encontrados!")
        st.info("Execute: `python main.py` para gerar os dados")
        return
    
    # Status dos algoritmos
    show_algorithm_status(data)
    
    # Métricas básicas
    st.subheader("📈 Métricas Principais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_files = len([k for k in data.keys() if isinstance(data[k], pd.DataFrame)])
        st.metric("📁 Arquivos", total_files)
    
    with col2:
        if 'original' in data:
            records = len(data['original'])
            st.metric("📋 Registros", f"{records:,}")
        else:
            st.metric("📋 Registros", "N/A")
    
    with col3:
        dbscan_count = len(data.get('dbscan_results', [])) if 'dbscan_results' in data else 0
        st.metric("🎯 Clusters", dbscan_count)
    
    with col4:
        rules_count = sum(len(data[k]) for k in ['apriori_rules', 'fp_growth_rules', 'eclat_rules'] if k in data)
        st.metric("🔗 Regras", rules_count)
    
    # Análise dos dados originais
    if 'original' in data:
        st.subheader("📊 Análise do Dataset")
        df = data['original']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Informações básicas:**")
            st.write(f"- Linhas: {len(df):,}")
            st.write(f"- Colunas: {len(df.columns)}")
            st.write(f"- Valores nulos: {df.isnull().sum().sum():,}")
        
        with col2:
            if 'salary' in df.columns:
                high_salary = (df['salary'] == '>50K').sum()
                low_salary = (df['salary'] == '<=50K').sum()
                st.write("**Distribuição salarial:**")
                st.write(f"- Salário >50K: {high_salary:,} ({high_salary/len(df)*100:.1f}%)")
                st.write(f"- Salário ≤50K: {low_salary:,} ({low_salary/len(df)*100:.1f}%)")
        
        # Gráfico de distribuição salarial
        if 'salary' in df.columns:
            st.subheader("📈 Distribuição Salarial")
            salary_counts = df['salary'].value_counts()
            
            fig = px.pie(
                values=salary_counts.values,
                names=salary_counts.index,
                title="Distribuição de Salários"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_clustering_page(data):
    """Página de clustering"""
    st.title("🎯 Clustering DBSCAN")
    
    if 'dbscan_results' not in data:
        st.warning("❌ Resultados DBSCAN não encontrados")
        st.info("Execute `python main.py` para gerar os resultados")
        return
    
    dbscan_df = data['dbscan_results']
    
    st.write("**Análise de Clustering baseada em densidade (DBSCAN)**")
    
    if 'cluster' in dbscan_df.columns:
        cluster_counts = dbscan_df['cluster'].value_counts().sort_index()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_clusters = len(cluster_counts[cluster_counts.index != -1])
            st.metric("🎯 Clusters", n_clusters)
        
        with col2:
            noise_points = cluster_counts.get(-1, 0)
            st.metric("🔴 Ruído", noise_points)
        
        with col3:
            noise_rate = (noise_points / len(dbscan_df)) * 100
            st.metric("📊 Taxa Ruído", f"{noise_rate:.1f}%")
        
        # Gráfico de distribuição
        st.subheader("📊 Distribuição dos Clusters")
        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            title="Número de Pontos por Cluster",
            labels={'x': 'Cluster ID', 'y': 'Número de Pontos'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela detalhada
        st.subheader("📋 Detalhes dos Clusters")
        cluster_details = []
        for cluster_id in sorted(cluster_counts.index):
            cluster_details.append({
                'Cluster': cluster_id,
                'Pontos': cluster_counts[cluster_id],
                'Percentual': f"{(cluster_counts[cluster_id] / len(dbscan_df)) * 100:.2f}%",
                'Tipo': 'Ruído' if cluster_id == -1 else 'Válido'
            })
        
        st.dataframe(pd.DataFrame(cluster_details), use_container_width=True)

# =============================================================================
# SIDEBAR MODERNIZADA MAS SIMPLES
# =============================================================================

def create_modern_sidebar():
    """Criar sidebar moderna mas simples"""
    
    # CSS para sidebar com botões estilizados
    st.markdown("""
    <style>
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        color: white;
    }
    
    .sidebar-info {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    
    .status-online { color: #28a745; }
    .status-offline { color: #dc3545; }
    
    /* Estilos para botões de navegação */
    .stButton > button {
        width: 100%;
        height: 3.5rem;
        background: linear-gradient(135deg, #4285f4 0%, #1976d2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        margin: 0.3rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(66, 133, 244, 0.3);
        border: 2px solid transparent;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #3367d6 0%, #1565c0 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(66, 133, 244, 0.4);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 10px rgba(66, 133, 244, 0.3);
    }
    
    .stButton > button:focus {
        outline: none;
        border-color: #ffffff;
        box-shadow: 0 0 0 3px rgba(66, 133, 244, 0.3);
    }
    
    /* Botão ativo */
    .active-nav-button > button {
        background: linear-gradient(135deg, #1976d2 0%, #0d47a1 100%) !important;
        box-shadow: 0 6px 20px rgba(25, 118, 210, 0.5) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Botão de logout especial */
    .logout-button > button {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%) !important;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.3) !important;
    }
    
    .logout-button > button:hover {
        background: linear-gradient(135deg, #c82333 0%, #a71e2a 100%) !important;
        box-shadow: 0 6px 20px rgba(220, 53, 69, 0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header do usuário
    username = st.session_state.get('username', 'User')
    role = st.session_state.get('user_role', 'User')
    
    st.sidebar.markdown(f"""
    <div class="sidebar-header">
        <h3 style="margin: 0;">👤 {username.title()}</h3>
        <p style="margin: 0.5rem 0; opacity: 0.9;">{role}</p>
        <small style="opacity: 0.8;">Online • {datetime.now().strftime('%H:%M')}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Navegação com botões
    st.sidebar.markdown("### 🧭 Navegação")
    
    # Inicializar página selecionada se não existir
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = 'overview'
    
    pages = [
        {"name": "📊 Visão Geral", "key": "overview", "description": "Dashboard principal"},
        {"name": "🎯 Clustering", "key": "clustering", "description": "Análise DBSCAN"},
        {"name": "🔗 Regras de Associação", "key": "rules", "description": "APRIORI, FP-GROWTH, ECLAT"},
        {"name": "🔮 Predição", "key": "prediction", "description": "ML Prediction System"}
    ]
    
    # Criar botões de navegação
    for page in pages:
        # Aplicar classe CSS especial se for a página ativa
        button_class = "active-nav-button" if st.session_state.selected_page == page["key"] else ""
        
        if button_class:
            st.sidebar.markdown(f'<div class="{button_class}">', unsafe_allow_html=True)
        
        if st.sidebar.button(
            page["name"], 
            key=f"nav_{page['key']}", 
            help=page["description"],
            use_container_width=True
        ):
            st.session_state.selected_page = page["key"]
            st.rerun()
        
        if button_class:
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Espaçamento
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # Status do sistema
    st.sidebar.markdown("### 📊 Status do Sistema")
    
    data = load_data()
    
    # Indicadores de status em cards compactos
    algorithms_status = {
        "🎯 DBSCAN": ("dbscan_results" in data and len(data["dbscan_results"]) > 0),
        "⛏️ APRIORI": ("apriori_rules" in data and len(data["apriori_rules"]) > 0),
        "🌳 FP-GROWTH": ("fp_growth_rules" in data and len(data["fp_growth_rules"]) > 0),
        "📊 ECLAT": ("eclat_rules" in data and len(data["eclat_rules"]) > 0)
    }
    
    for alg, status in algorithms_status.items():
        status_color = "#28a745" if status else "#dc3545"
        status_text = "Ativo" if status else "Inativo"
        status_icon = "🟢" if status else "🔴"
        
        st.sidebar.markdown(f"""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.3rem 0;
            border-left: 3px solid {status_color};
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <span style="font-weight: 500;">{alg}</span>
            <span style="color: {status_color}; font-size: 0.9rem;">
                {status_icon} {status_text}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Métricas rápidas em cards
    st.sidebar.markdown("### 📈 Métricas Rápidas")
    
    if data:
        total_files = len(data)
        
        # Card de arquivos
        st.sidebar.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #4285f4, #1976d2);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 4px 15px rgba(66, 133, 244, 0.3);
        ">
            <div style="font-size: 1.8rem; margin-bottom: 0.3rem;">📁</div>
            <div style="font-size: 1.4rem; font-weight: bold;">{total_files}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Arquivos Carregados</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Card de registros
        if 'original' in data:
            records = len(data['original'])
            st.sidebar.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #28a745, #20c997);
                color: white;
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                margin: 0.5rem 0;
                box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
            ">
                <div style="font-size: 1.8rem; margin-bottom: 0.3rem;">📋</div>
                <div style="font-size: 1.4rem; font-weight: bold;">{records:,}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Registros Totais</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Calcular total de regras
        total_rules = sum(len(data[k]) for k in ['apriori_rules', 'fp_growth_rules', 'eclat_rules'] if k in data)
        if total_rules > 0:
            st.sidebar.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #ffc107, #e0a800);
                color: white;
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                margin: 0.5rem 0;
                box-shadow: 0 4px 15px rgba(255, 193, 7, 0.3);
            ">
                <div style="font-size: 1.8rem; margin-bottom: 0.3rem;">🔗</div>
                <div style="font-size: 1.4rem; font-weight: bold;">{total_rules}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Regras Geradas</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Card de aviso
        st.sidebar.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #dc3545, #c82333);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 4px 15px rgba(220, 53, 69, 0.3);
        ">
            <div style="font-size: 1.8rem; margin-bottom: 0.3rem;">⚠️</div>
            <div style="font-size: 1rem; font-weight: bold;">Dados Não Encontrados</div>
            <div style="font-size: 0.8rem; opacity: 0.9;">Execute: python main.py</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Ações rápidas
    st.sidebar.markdown("### ⚡ Ações Rápidas")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄", key="refresh", help="Recarregar Dados", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("📥", key="export", help="Exportar Resultados", use_container_width=True):
            st.sidebar.success("Em desenvolvimento")
    
    # Logout com estilo especial
    st.sidebar.markdown("### 🚪 Sessão")
    
    st.sidebar.markdown('<div class="logout-button">', unsafe_allow_html=True)
    if st.sidebar.button("🚪 Logout Seguro", key="logout", use_container_width=True):
        # Limpar estado da sessão
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.session_state.username = None
        st.session_state.selected_page = 'overview'
        st.rerun()
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Info técnica compacta
    with st.sidebar.expander("ℹ️ Informações Técnicas"):
        st.markdown(f"""
        **🕒 Sessão Atual:**
        - Data: {datetime.now().strftime('%d/%m/%Y')}
        - Hora: {datetime.now().strftime('%H:%M:%S')}
        - Streamlit: {st.__version__}
        
        **🔧 Sistema:**
        - Página Ativa: {st.session_state.selected_page}
        - Usuário: {username}
        - Role: {role}
        """)
    
    return st.session_state.selected_page

# =============================================================================
# APLICAÇÃO PRINCIPAL
# =============================================================================

def main():
    """Aplicação principal simplificada"""
    
    # Inicializar estado
    init_session_state()
    
    # CSS global para a aplicação
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    
    /* Melhorar espaçamento geral */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Verificar login
    if not st.session_state.logged_in:
        show_login()
        return
    
    # Carregar dados
    data = load_data()
    
    # Criar sidebar e obter página selecionada
    selected_page = create_modern_sidebar()
    
    # Roteamento das páginas
    try:
        if selected_page == "overview":
            show_overview_page(data)
        elif selected_page == "clustering":
            show_clustering_page(data)
        elif selected_page == "rules":
            # Usar a nova função importada
            show_association_rules_page(data)
        elif selected_page == "prediction":
            show_prediction_page(data)
        else:
            st.error("Página não encontrada")
    except Exception as e:
        st.error(f"Erro ao carregar página: {e}")
        st.info("Tente recarregar a página ou contacte o suporte")
        
        # Debug info para desenvolvimento
        if st.session_state.get('username') == 'admin':
            with st.expander("🔧 Debug Info (Admin)"):
                st.code(str(e))
                st.write("**Dados disponíveis:**", list(data.keys()) if data else "Nenhum")
    
    # Footer simples
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎓 Dashboard Acadêmico - Análise Salarial Científica</p>
        <p><small>DBSCAN • APRIORI • FP-GROWTH • ECLAT</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()