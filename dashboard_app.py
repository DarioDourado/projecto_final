import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import hashlib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# ======================================
# 0. SISTEMA DE AUTENTICAÇÃO - PRIMEIRA COISA
# ======================================

def hash_password(password):
    """Criar hash da password para segurança"""
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_password():
    """Verificar credenciais do utilizador"""
    
    # Definir utilizadores e passwords (em produção, usar base de dados)
    users_db = {
        "admin": hash_password("admin123"),
        "user": hash_password("user123"),
        "analista": hash_password("dados2024"),
        "demo": hash_password("demo")
    }
    
    def password_entered():
        """Verificar se username/password estão corretos"""
        username = st.session_state.get("username", "")
        password = st.session_state.get("password", "")
        
        if username in users_db and users_db[username] == hash_password(password):
            st.session_state["password_correct"] = True
            st.session_state["current_user"] = username
            # Limpar password da sessão por segurança
            if "password" in st.session_state:
                del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # Se ainda não autenticado
    if "password_correct" not in st.session_state:
        # Configurar página de login
        st.set_page_config(
            page_title="Login - Análise Salarial", 
            layout="centered",
            page_icon="🔐"
        )
        
        # Mostrar tela de login
        st.markdown("# 🔐 Login - Dashboard Análise Salarial")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 👤 Autenticação Necessária")
            st.text_input("👤 Utilizador", key="username", placeholder="Digite seu username")
            st.text_input("🔑 Password", type="password", key="password", placeholder="Digite sua password")
            
            if st.button("🚀 Entrar", type="primary", use_container_width=True):
                password_entered()
            
            # Mostrar credenciais de demonstração
            with st.expander("🔍 Credenciais de Demonstração"):
                st.markdown("""
                **Contas disponíveis para teste:**
                
                | Utilizador | Password | Nível de Acesso |
                |------------|----------|-----------------|
                | `admin` | `admin123` | 👑 Administrador (Acesso Total) |
                | `analista` | `dados2024` | 📊 Analista (Análises + Previsões) |
                | `user` | `user123` | 👤 Utilizador (Básico) |
                | `demo` | `demo` | 🔍 Demo (Apenas Visualização) |
                """)
        
        st.stop()  # Para a execução aqui se não estiver autenticado
    
    # Se password incorreta
    elif not st.session_state["password_correct"]:
        st.set_page_config(
            page_title="Login - Análise Salarial", 
            layout="centered",
            page_icon="❌"
        )
        
        st.error("❌ Username ou password incorretos!")
        st.markdown("# 🔐 Login - Dashboard Análise Salarial")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("👤 Utilizador", key="username", placeholder="Digite seu username")
            st.text_input("🔑 Password", type="password", key="password", placeholder="Digite sua password")
            
            if st.button("🚀 Entrar", type="primary", use_container_width=True):
                password_entered()
        
        st.stop()  # Para a execução aqui se a password estiver incorreta
    
    # Se autenticado com sucesso, continuar com o dashboard
    else:
        return True

def logout():
    """Fazer logout do utilizador"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def get_user_permissions(username):
    """Definir permissões por utilizador"""
    permissions = {
        "admin": {
            "view_data": True,
            "view_visualizations": True,
            "view_models": True,
            "make_predictions": True,
            "upload_files": True,
            "view_system_info": True
        },
        "analista": {
            "view_data": True,
            "view_visualizations": True,
            "view_models": True,
            "make_predictions": True,
            "upload_files": True,
            "view_system_info": False
        },
        "user": {
            "view_data": True,
            "view_visualizations": True,
            "view_models": False,
            "make_predictions": True,
            "upload_files": False,
            "view_system_info": False
        },
        "demo": {
            "view_data": True,
            "view_visualizations": True,
            "view_models": False,
            "make_predictions": False,
            "upload_files": False,
            "view_system_info": False
        }
    }
    return permissions.get(username, permissions["demo"])

# ======================================
# 1. VERIFICAR AUTENTICAÇÃO ANTES DE CONTINUAR
# ======================================
if not check_password():
    st.stop()

# ======================================
# 2. CONFIGURAÇÃO INICIAL DO STREAMLIT (APÓS LOGIN)
# ======================================
st.set_page_config(
    page_title="Dashboard - Análise Salarial", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Obter permissões do utilizador
user_permissions = get_user_permissions(st.session_state['current_user'])

# Mostrar utilizador logado e botão de logout no sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown(f"👤 **Utilizador:** `{st.session_state['current_user']}`")
    
    # Mostrar nível de acesso
    access_levels = {
        "admin": "👑 Administrador",
        "analista": "📊 Analista", 
        "user": "👤 Utilizador",
        "demo": "🔍 Demo"
    }
    st.markdown(f"🎯 **Nível:** {access_levels.get(st.session_state['current_user'], '🔍 Demo')}")
    
    if st.button("🚪 Logout", type="secondary", use_container_width=True):
        logout()
    st.markdown("---")

# Importação opcional do SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.title("📊 Dashboard Interativo - Análise e Previsão Salarial")
st.markdown(f"🎯 **Bem-vindo(a), {st.session_state['current_user']}!**")

# Mostrar aviso sobre SHAP apenas se não estiver disponível
if not SHAP_AVAILABLE:
    st.warning("⚠️ Pacote SHAP não instalado. Algumas funcionalidades podem não estar disponíveis.")

# ======================================
# 2.5 SISTEMA DE TRADUÇÃO E TIPAGEM PARA INTERFACE
# ======================================
def get_translations():
    """Dicionário de traduções para a interface"""
    return {
        # Workclass
        'Private': 'Empresa Privada',
        'Self-emp-not-inc': 'Autônomo (Não Incorporado)',
        'Self-emp-inc': 'Autônomo (Incorporado)',
        'Federal-gov': 'Governo Federal',
        'Local-gov': 'Governo Local',
        'State-gov': 'Governo Estadual',
        'Without-pay': 'Sem Remuneração',
        'Never-worked': 'Nunca Trabalhou',
        
        # Education
        'Preschool': 'Pré-escolar',
        '1st-4th': '1º ao 4º ano',
        '5th-6th': '5º ao 6º ano',
        '7th-8th': '7º ao 8º ano',
        '9th': '9º ano',
        '10th': '10º ano',
        '11th': '11º ano',
        '12th': '12º ano',
        'HS-grad': 'Ensino Médio Completo',
        'Some-college': 'Superior Incompleto',
        'Assoc-voc': 'Curso Técnico',
        'Assoc-acdm': 'Curso Técnico Superior',
        'Bachelors': 'Licenciatura',
        'Masters': 'Mestrado',
        'Prof-school': 'Escola Profissional',
        'Doctorate': 'Doutoramento',
        
        # Marital Status
        'Married-civ-spouse': 'Casado(a)',
        'Divorced': 'Divorciado(a)',
        'Never-married': 'Solteiro(a)',
        'Separated': 'Separado(a)',
        'Widowed': 'Viúvo(a)',
        'Married-spouse-absent': 'Casado(a) - Cônjuge Ausente',
        'Married-AF-spouse': 'Casado(a) - Militar',
        
        # Occupation
        'Tech-support': 'Suporte Técnico',
        'Craft-repair': 'Artesanato/Reparação',
        'Other-service': 'Outros Serviços',
        'Sales': 'Vendas',
        'Exec-managerial': 'Executivo/Gestão',
        'Prof-specialty': 'Profissional Especializado',
        'Handlers-cleaners': 'Manipulação/Limpeza',
        'Machine-op-inspct': 'Operador de Máquinas',
        'Adm-clerical': 'Administrativo',
        'Farming-fishing': 'Agricultura/Pesca',
        'Transport-moving': 'Transporte/Mudanças',
        'Priv-house-serv': 'Serviços Domésticos',
        'Protective-serv': 'Serviços de Proteção',
        'Armed-Forces': 'Forças Armadas',
        
        # Relationship
        'Wife': 'Esposa',
        'Own-child': 'Filho(a)',
        'Husband': 'Marido',
        'Not-in-family': 'Não é Família',
        'Other-relative': 'Outro Parente',
        'Unmarried': 'Não Casado(a)',
        
        # Race
        'White': 'Branco',
        'Asian-Pac-Islander': 'Asiático',
        'Amer-Indian-Eskimo': 'Indígena Americano',
        'Other': 'Outro',
        'Black': 'Negro',
        
        # Sex
        'Male': 'Masculino',
        'Female': 'Feminino',
        
        # Native Country
        'United-States': 'Estados Unidos',
        'Canada': 'Canadá',
        'Germany': 'Alemanha',
        'India': 'Índia',
        'Japan': 'Japão',
        'Mexico': 'México',
        'Philippines': 'Filipinas',
        'Puerto-Rico': 'Porto Rico',
        'El-Salvador': 'El Salvador',
        'Cuba': 'Cuba',
        'England': 'Inglaterra',
        'China': 'China',
        'Italy': 'Itália',
        'Poland': 'Polônia',
        'Vietnam': 'Vietnã',
        
        # Salary
        '<=50K': 'Até 50K',
        '>50K': 'Acima de 50K'
    }

def get_column_info():
    """Informações sobre tipos e validações das colunas"""
    return {
        'age': {
            'type': 'int16',
            'min': 17,
            'max': 100,
            'description': 'Idade da pessoa',
            'unit': 'anos'
        },
        'fnlwgt': {
            'type': 'int32',
            'min': 1,
            'max': 1500000,
            'description': 'Peso final (ponderação demográfica)',
            'unit': 'peso estatístico'
        },
        'education-num': {
            'type': 'int8',
            'min': 1,
            'max': 16,
            'description': 'Número de anos de educação formal',
            'unit': 'anos'
        },
        'capital-gain': {
            'type': 'int32',
            'min': 0,
            'max': 100000,
            'description': 'Ganhos de capital',
            'unit': 'dólares'
        },
        'capital-loss': {
            'type': 'int16',
            'min': 0,
            'max': 5000,
            'description': 'Perdas de capital',
            'unit': 'dólares'
        },
        'hours-per-week': {
            'type': 'int8',
            'min': 1,
            'max': 99,
            'description': 'Horas trabalhadas por semana',
            'unit': 'horas'
        }
    }

def validate_and_convert_input(value, column_name):
    """Validar e converter entrada do utilizador"""
    column_info = get_column_info()
    
    if column_name in column_info:
        info = column_info[column_name]
        
        # Converter para o tipo correto
        if info['type'].startswith('int'):
            try:
                value = int(value)
            except (ValueError, TypeError):
                st.error(f"❌ {column_name} deve ser um número inteiro")
                return None
        
        # Validar range
        if value < info['min'] or value > info['max']:
            st.warning(f"⚠️ {column_name} deve estar entre {info['min']} e {info['max']} {info['unit']}")
            return None
    
    return value

def translate_value(value, translations=None):
    """Traduzir um valor usando o dicionário de traduções"""
    if translations is None:
        translations = get_translations()
    return translations.get(str(value), str(value))

def translate_column(series, translations=None):
    """Traduzir uma coluna inteira"""
    if translations is None:
        translations = get_translations()
    return series.apply(lambda x: translate_value(x, translations))

def get_reverse_translations():
    """Criar dicionário reverso para converter de português para inglês"""
    translations = get_translations()
    return {v: k for k, v in translations.items()}

@st.cache_data
def load_data():
    """Carregar dados com tratamento de erros robusto"""
    try:
        print("🔄 Tentando carregar dados...")
        
        # Verificar se o arquivo existe
        if not os.path.exists("4-Carateristicas_salario.csv"):
            print("❌ Arquivo não encontrado!")
            return pd.DataFrame()
        
        # Carregar dados
        df = pd.read_csv("4-Carateristicas_salario.csv")
        
        # Verificar se o arquivo não está vazio
        if df.empty:
            print("❌ Arquivo está vazio!")
            return pd.DataFrame()
        
        # Limpar nomes das colunas
        df.columns = df.columns.str.strip()
        
        # Remover duplicatas
        original_size = len(df)
        df = df.drop_duplicates()
        removed_duplicates = original_size - len(df)
        
        if removed_duplicates > 0:
            print(f"🧹 Removidas {removed_duplicates} linhas duplicadas")
        
        # Aplicar tipagem automática se os dados não estiverem tipados
        print("🔧 Verificando e aplicando tipagem...")
        
        # Definir tipos esperados
        expected_types = {
            'age': 'int16',
            'fnlwgt': 'int32',
            'education-num': 'int8',
            'capital-gain': 'int32',
            'capital-loss': 'int16',
            'hours-per-week': 'int8'
        }
        
        for col, expected_type in expected_types.items():
            if col in df.columns:
                try:
                    # Verificar se já está no tipo correto
                    if df[col].dtype != expected_type:
                        # Converter primeiro para numérico, depois para o tipo específico
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col] = df[col].astype(expected_type)
                        print(f"✅ {col}: convertido para {expected_type}")
                    else:
                        print(f"✅ {col}: já é {expected_type}")
                except Exception as e:
                    print(f"⚠️ Não foi possível converter {col} para {expected_type}: {e}")
        
        # Converter categóricas para category se ainda não estiverem
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                           'relationship', 'race', 'sex', 'native-country', 'salary']
        
        for col in categorical_cols:
            if col in df.columns:
                try:
                    if df[col].dtype != 'category':
                        df[col] = df[col].astype('category')
                        print(f"✅ {col}: convertido para category")
                    else:
                        print(f"✅ {col}: já é category")
                except Exception as e:
                    print(f"⚠️ Não foi possível converter {col} para category: {e}")
        
        print(f"✅ Dados carregados com sucesso! Shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        print("❌ Arquivo '4-Carateristicas_salario.csv' não encontrado!")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print("❌ Arquivo CSV está vazio!")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        print(f"❌ Erro ao analisar CSV: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Erro inesperado ao carregar dados: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        model = joblib.load("random_forest_model.joblib")
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {str(e)}")
        return None

@st.cache_resource
def load_preprocessor():
    try:
        preprocessor = joblib.load("preprocessor.joblib")
        return preprocessor
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"❌ Erro ao carregar preprocessor: {str(e)}")
        return None

@st.cache_data
def load_feature_info():
    try:
        return joblib.load("feature_info.joblib")
    except FileNotFoundError:
        return None

# Carregar recursos LOGO APÓS VERIFICAÇÃO DE AUTENTICAÇÃO
print("🔄 Carregando dados...")
df = load_data()
print("🔄 Carregando modelo...")
model = load_model()
print("🔄 Carregando preprocessor...")
preprocessor = load_preprocessor()
print("🔄 Carregando informações das features...")
feature_info = load_feature_info()

# Verificar se os dados foram carregados com sucesso
if df.empty:
    st.error("❌ Não foi possível carregar os dados!")
    st.markdown("""
    **Possíveis soluções:**
    1. Verifique se o arquivo `4-Carateristicas_salario.csv` existe no diretório
    2. Execute primeiro: `python projeto_salario.py`
    3. Contacte o administrador se o problema persistir
    """)
    st.stop()

# ======================================
# 3. VERIFICAR ARQUIVOS DO SISTEMA
# ======================================
if user_permissions["view_system_info"]:
    st.markdown("### 🔍 Status do Sistema")
    
    required_files = {
        "Dados": {
            "file": "4-Carateristicas_salario.csv",
            "icon": "📊",
            "description": "Dataset principal com características salariais",
            "critical": True
        },
        "Modelo": {
            "file": "random_forest_model.joblib",
            "icon": "🤖",
            "description": "Modelo Random Forest treinado",
            "critical": True
        },
        "Preprocessor": {
            "file": "preprocessor.joblib",
            "icon": "⚙️",
            "description": "Pipeline de pré-processamento dos dados",
            "critical": True
        },
        "Feature Info": {
            "file": "feature_info.joblib",
            "icon": "📋",
            "description": "Informações sobre as features do modelo",
            "critical": False
        },
        "Gráficos": {
            "file": "imagens",
            "icon": "📈",
            "description": "Pasta com gráficos gerados",
            "critical": False
        }
    }

    # Container principal com estilo moderno
    with st.container():
        # Cabeçalho da seção
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("#### 🔍 Verificação de Componentes")
        with col2:
            if st.button("🔄 Atualizar", help="Verificar novamente o status"):
                st.rerun()
        
        st.markdown("---")
        
        # Verificar status de cada arquivo
        file_status = {}
        all_critical_ok = True
        total_files = len(required_files)
        files_ok = 0
        
        for name, info in required_files.items():
            if info["file"] == "imagens":
                exists = os.path.exists("imagens") and os.path.isdir("imagens")
                size_info = ""
                if exists:
                    try:
                        files_in_imagens = len([f for f in os.listdir("imagens") if f.endswith(('.png', '.jpg', '.jpeg', '.svg'))])
                        size_info = f"{files_in_imagens} gráficos"
                    except:
                        size_info = "Não acessível"
            else:
                exists = os.path.exists(info["file"])
                size_info = ""
                if exists:
                    try:
                        size = os.path.getsize(info["file"])
                        if size < 1024:
                            size_info = f"{size} B"
                        elif size < 1024**2:
                            size_info = f"{size/1024:.1f} KB"
                        elif size < 1024**3:
                            size_info = f"{size/(1024**2):.1f} MB"
                        else:
                            size_info = f"{size/(1024**3):.1f} GB"
                    except:
                        size_info = "Tamanho indefinido"
            
            file_status[name] = {
                "exists": exists,
                "size": size_info,
                "critical": info["critical"]
            }
            
            if exists:
                files_ok += 1
            elif info["critical"]:
                all_critical_ok = False
        
        # Status dos componentes de forma simples
        st.markdown("#### 🔍 Status dos Componentes")
        
        # 5 Mini Cards simples
        st.markdown("#### 🔍 Status dos Componentes")
        
        # Criar linha com 5 colunas
        col1, col2, col3, col4, col5 = st.columns(5)
        columns = [col1, col2, col3, col4, col5]
        
        for idx, (name, info) in enumerate(required_files.items()):
            exists = file_status[name]["exists"]
            size_info = file_status[name]["size"]
            is_critical = file_status[name]["critical"]
            
            with columns[idx]:
                # Card container
                st.markdown(f"""
                <div style="
                    text-align: center;
                    padding: 15px;
                    border-radius: 10px;
                    border: 2px solid {'#28a745' if exists else '#dc3545' if is_critical else '#ffc107'};
                    background-color: {'#d4edda' if exists else '#f8d7da' if is_critical else '#fff3cd'};
                    margin-bottom: 10px;
                ">
                    <div style="font-size: 30px; margin-bottom: 8px;">{info['icon']}</div>
                    <div style="font-weight: bold; margin-bottom: 5px;">{name}</div>
                    <div style="font-size: 20px;">{'✅' if exists else '❌' if is_critical else '⚠️'}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Informações adicionais
                if size_info:
                    st.caption(f"📦 {size_info}")
                
                # Estatísticas rápidas do sistema
                with st.expander("📊 Estatísticas Detalhadas", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if not df.empty:
                            st.metric("📊 Registos no Dataset", f"{len(df):,}")
                            st.metric("📋 Colunas Disponíveis", len(df.columns))
                    
                    with col2:
                        if os.path.exists("imagens"):
                            img_files = len([f for f in os.listdir("imagens") if f.endswith(('.png', '.jpg', '.jpeg', '.svg'))])
                            st.metric("📈 Gráficos Gerados", img_files)
                        
                        if model is not None:
                            st.metric("🤖 Modelos Carregados", "1")
                    
                    with col3:
                        try:
                            import psutil
                            memory_usage = psutil.virtual_memory().percent
                            st.metric("💾 Uso de Memória", f"{memory_usage:.1f}%")
                        except ImportError:
                            st.metric("💾 Sistema", "✅ Ativo")
                        except Exception:
                            st.metric("💾 Sistema", "✅ Ativo")

# ======================================
# 4. STATUS DOS MODELOS E COMPONENTES
# ======================================
# Mostrar status dos modelos (apenas para utilizadores com permissão)
if user_permissions["view_models"]:
    col1, col2 = st.columns(2)
    
    with col1:
        if model is not None:
            st.success("✅ Modelo Random Forest carregado com sucesso!")
        else:
            st.warning("⚠️ Modelo não encontrado. Execute: `python projeto_salario.py`")
    
    with col2:
        if preprocessor is not None:
            st.success("✅ Preprocessor carregado com sucesso!")
        else:
            st.warning("⚠️ Preprocessor não encontrado.")

# Tratar avisos do sklearn sobre categorias desconhecidas
if model is not None and preprocessor is not None:
    st.info("ℹ️ Os avisos sobre 'unknown categories' são normais durante previsões com dados novos e não afetam o funcionamento.")

# ======================================
# 4. VISÃO GERAL DOS DADOS
# ======================================
if user_permissions["view_data"]:
    st.sidebar.header("🔧 Filtros")

    # Verificar se as colunas existem antes de criar filtros
    available_columns = df.columns.tolist()

    if "sex" in available_columns:
        sexo = st.sidebar.selectbox("Sexo", options=["Todos"] + sorted(df["sex"].unique()))
    else:
        sexo = "Todos"
        st.sidebar.warning("Coluna 'sex' não encontrada")

    if "education" in available_columns:
        educacao = st.sidebar.selectbox("Educação", options=["Todos"] + sorted(df["education"].unique()))
    else:
        educacao = "Todos"
        st.sidebar.warning("Coluna 'education' não encontrada")

    if "native-country" in available_columns:
        pais = st.sidebar.selectbox("País de Origem", options=["Todos"] + sorted(df["native-country"].unique()))
    else:
        pais = "Todos"
        st.sidebar.warning("Coluna 'native-country' não encontrada")

    # Aplicar filtros
    df_filtrado = df.copy()
    if sexo != "Todos" and "sex" in available_columns:
        df_filtrado = df_filtrado[df_filtrado["sex"] == sexo]
    if educacao != "Todos" and "education" in available_columns:
        df_filtrado = df_filtrado[df_filtrado["education"] == educacao]
    if pais != "Todos" and "native-country" in available_columns:
        df_filtrado = df_filtrado[df_filtrado["native-country"] == pais]

    st.markdown("### 📌 Estatísticas Gerais do Conjunto de Dados")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Registos", len(df_filtrado))

    if "salary" in available_columns:
        # Contar ambas as variações possíveis de salary
        baixo_salario = (df_filtrado["salary"].str.strip() == "<=50K").sum()
        alto_salario = (df_filtrado["salary"].str.strip() == ">50K").sum()
        
        col2.metric("Classe <=50K", baixo_salario)
        col3.metric("Classe >50K", alto_salario)
    else:
        col2.metric("Classe <=50K", "N/A")
        col3.metric("Classe >50K", "N/A")
else:
    df_filtrado = df

# ======================================
# 5. GRÁFICOS INTERATIVOS - ATUALIZAR CAMINHOS
# ======================================
if user_permissions["view_visualizations"]:
    st.markdown("### 📈 Visualizações")

    # Criar tabs baseadas nas permissões
    available_tabs = ["📊 Distribuições"]
    
    if user_permissions["view_models"]:
        available_tabs.extend(["🔍 Importância", "🎯 Clustering"])
    
    available_tabs.append("📈 Correlações")
    
    tabs = st.tabs(available_tabs)

    with tabs[0]:  # Distribuições
        st.markdown("#### Distribuições das Variáveis")
        
        # ✅ CORRIGIR: Todos os caminhos agora apontam para pasta imagens/
        image_files = {
            "Distribuição de Idade": "imagens/hist_age.png",
            "Distribuição de Educação": "imagens/hist_education-num.png", 
            "Distribuição de Horas/Semana": "imagens/hist_hours-per-week.png",
            "Distribuição de Ganho Capital": "imagens/hist_capital-gain.png",
            "Distribuição de Perda Capital": "imagens/hist_capital-loss.png",
            "Distribuição de Peso Final": "imagens/hist_fnlwgt.png"
        }
        
        # Verificar se a pasta imagens existe
        if not os.path.exists("imagens"):
            st.warning("📁 Pasta 'imagens' não encontrada. Execute primeiro: `python projeto_salario.py`")
        else:
            # Mostrar gráficos existentes em grid 2x3
            cols = st.columns(2)
            for idx, (title, file_path) in enumerate(image_files.items()):
                try:
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        with cols[idx % 2]:
                            st.image(file_path, caption=title, use_column_width=True)
                    else:
                        with cols[idx % 2]:
                            st.info(f"⏳ {title} ainda não gerado")
                except Exception as e:
                    cols[idx % 2].error(f"❌ Erro ao carregar {title}: {str(e)}")

    if user_permissions["view_models"] and len(tabs) > 1:
        with tabs[1]:  # Importância das Features
            st.markdown("#### 🔍 Importância das Features")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Random Forest - Importância**")
                try:
                    rf_importance_path = "imagens/feature_importance_rf.png"
                    if os.path.exists(rf_importance_path) and os.path.getsize(rf_importance_path) > 0:
                        st.image(rf_importance_path, caption="Importância das Features - Random Forest", use_column_width=True)
                    else:
                        st.info("⏳ Gráfico de importância RF ainda não gerado")
                except Exception as e:
                    st.error(f"❌ Erro ao carregar importância RF: {str(e)}")
            
            with col2:
                st.markdown("**Regressão Logística - Coeficientes**")
                try:
                    lr_coef_path = "imagens/coefficients_lr.png"
                    if os.path.exists(lr_coef_path) and os.path.getsize(lr_coef_path) > 0:
                        st.image(lr_coef_path, caption="Coeficientes - Regressão Logística", use_column_width=True)
                    else:
                        st.info("⏳ Gráfico de coeficientes LR ainda não gerado")
                except Exception as e:
                    st.error(f"❌ Erro ao carregar coeficientes LR: {str(e)}")

        with tabs[2]:  # Clustering e Análises Avançadas
            st.markdown("#### 🎯 Análise de Clustering e Visualizações Avançadas")
            
            # Clustering em duas colunas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Clustering K-Means**")
                try:
                    kmeans_path = "imagens/kmeans_clusters.png"
                    if os.path.exists(kmeans_path) and os.path.getsize(kmeans_path) > 0:
                        st.image(kmeans_path, caption="Clusters K-Means (PCA 2D)", use_column_width=True)
                    else:
                        st.info("⏳ Gráfico de clustering ainda não gerado")
                except Exception as e:
                    st.error(f"❌ Erro ao carregar clustering: {str(e)}")
            
            with col2:
                st.markdown("**Análise de Componentes Principais**")
                try:
                    pca_path = "imagens/pca_analysis.png"
                    if os.path.exists(pca_path) and os.path.getsize(pca_path) > 0:
                        st.image(pca_path, caption="Análise PCA", use_column_width=True)
                    else:
                        st.info("⏳ Gráfico PCA ainda não gerado")
                except Exception as e:
                    st.error(f"❌ Erro ao carregar PCA: {str(e)}")
            
            # SHAP Analysis (se disponível)
            if SHAP_AVAILABLE:
                st.markdown("**Interpretabilidade SHAP**")
                shap_files = {
                    "SHAP Summary": "imagens/shap_summary.png",
                    "SHAP Waterfall": "imagens/shap_waterfall.png",
                    "SHAP Dependence": "imagens/shap_dependence.png"
                }
                
                shap_cols = st.columns(len(shap_files))
                for idx, (title, path) in enumerate(shap_files.items()):
                    with shap_cols[idx]:
                        try:
                            if os.path.exists(path) and os.path.getsize(path) > 0:
                                st.image(path, caption=title, use_column_width=True)
                            else:
                                st.info(f"⏳ {title} ainda não gerado")
                        except Exception as e:
                            st.error(f"❌ Erro ao carregar {title}: {str(e)}")

    with tabs[-1]:  # Correlações (sempre último)
        st.markdown("#### 📊 Matriz de Correlação")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            try:
                correlation_path = "imagens/correlacao.png"
                if os.path.exists(correlation_path) and os.path.getsize(correlation_path) > 0:
                    st.image(correlation_path, caption="Correlação entre Variáveis Numéricas", use_column_width=True)
                else:
                    st.info("⏳ Matriz de correlação ainda não gerada")
            except Exception as e:
                st.error(f"❌ Erro ao carregar matriz de correlação: {str(e)}")
        
        with col2:
            st.markdown("**Informações sobre Correlação:**")
            st.info("""
            🔍 **Como interpretar:**
            
            • **Cores quentes** (vermelho): Correlação positiva forte
            • **Cores frias** (azul): Correlação negativa forte  
            • **Branco**: Sem correlação
            
            📊 **Valores:**
            • +1.0: Correlação perfeita positiva
            • 0.0: Sem correlação
            • -1.0: Correlação perfeita negativa
            """)

# ======================================
# 6. ANÁLISE EXPLORATÓRIA INTERATIVA
# ======================================
if user_permissions["view_data"]:
    st.markdown("### 🔍 Análise Exploratória Interativa")

    if not df_filtrado.empty:
        # Seleção de variável para análise
        numerical_cols = df_filtrado.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_filtrado.select_dtypes(include=['object']).columns.tolist()
        
        if not numerical_cols and not categorical_cols:
            st.warning("⚠️ Nenhuma coluna disponível para análise")
        else:
            analysis_type = st.selectbox("Tipo de Análise", ["Distribuição Numérica", "Distribuição Categórica"])
            
            if analysis_type == "Distribuição Numérica" and numerical_cols:
                selected_col = st.selectbox("Selecione uma variável numérica", numerical_cols)
                
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Configurar fundo transparente
                    fig.patch.set_alpha(0.0)
                    ax.patch.set_alpha(0.0)
                    
                    # Verificar se a coluna tem dados válidos
                    data_valid = df_filtrado[selected_col].dropna()
                    if len(data_valid) > 0:
                        sns.histplot(data=df_filtrado, x=selected_col, kde=True, ax=ax)
                        ax.set_title(f"Distribuição de {selected_col}")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig, transparent=True)
                    else:
                        st.warning(f"⚠️ Não há dados válidos para {selected_col}")
                    plt.close()
                    
                except Exception as e:
                    st.error(f"❌ Erro ao gerar gráfico: {str(e)}")
                    plt.close()
                    
            elif analysis_type == "Distribuição Categórica" and categorical_cols:
                selected_col = st.selectbox("Selecione uma variável categórica", categorical_cols)
                
                try:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    # Configurar fundo transparente
                    fig.patch.set_alpha(0.0)
                    ax.patch.set_alpha(0.0)
                    
                    # Verificar se a coluna tem dados válidos
                    data_valid = df_filtrado[selected_col].dropna()
                    if len(data_valid) > 0:
                        value_counts = df_filtrado[selected_col].value_counts()
                        if len(value_counts) > 0:
                            value_counts.plot(kind='bar', ax=ax)
                            ax.set_title(f"Distribuição de {selected_col}")
                            ax.tick_params(axis='x', rotation=45)
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig, transparent=True)
                        else:
                            st.warning(f"⚠️ Não há dados para visualizar em {selected_col}")
                    else:
                        st.warning(f"⚠️ Não há dados válidos para {selected_col}")
                    plt.close()
                    
                except Exception as e:
                    st.error(f"❌ Erro ao gerar gráfico: {str(e)}")
                    plt.close()
            else:
                if analysis_type == "Distribuição Numérica":
                    st.info("ℹ️ Nenhuma variável numérica disponível")
                else:
                    st.info("ℹ️ Nenhuma variável categórica disponível")
    else:
        st.warning("⚠️ Nenhum dado disponível após aplicar filtros")

# ======================================
# 7. PREVISÃO DE NOVOS DADOS
# ======================================
if user_permissions["make_predictions"]:
    st.markdown("### 🔮 Previsão com Novos Dados")

    # Adicionar seção explicativa das variáveis
    with st.expander("📖 Dicionário de Dados - Significado das Variáveis"):
        st.markdown("""
        ### 📊 Descrição das Variáveis do Dataset
        
        **Variáveis Numéricas:**
        
        | Variável | Descrição | Valores |
        |----------|-----------|---------|
        | **Age** | Idade da pessoa | 17-90 anos |
        | **Education-num** | Número de anos de educação formal | 1-16 anos |
        | **Hours-per-week** | Horas trabalhadas por semana | 1-99 horas |
        | **Capital-gain** | Ganhos de capital (investimentos, ações, etc.) | 0+ dólares |
        | **Capital-loss** | Perdas de capital (investimentos, ações, etc.) | 0+ dólares |
        | **Fnlwgt** | Peso final (ponderação demográfica) | Peso estatístico |
        
        **Variáveis Categóricas:**
        
        **🏢 Workclass (Classe de Trabalho):**
        - **Private**: Empresa privada
        - **Self-emp-not-inc**: Trabalhador independente (não incorporado)
        - **Self-emp-inc**: Trabalhador independente (incorporado)
        - **Federal-gov**: Governo federal
        - **Local-gov**: Governo local
        - **State-gov**: Governo estadual
        - **Without-pay**: Sem remuneração
        - **Never-worked**: Nunca trabalhou
        
        **🎓 Education (Nível de Educação):**
        - **Preschool**: Pré-escolar
        - **1st-4th**: 1º ao 4º ano
        - **5th-6th**: 5º ao 6º ano
        - **7th-8th**: 7º ao 8º ano
        - **9th**: 9º ano
        - **10th**: 10º ano
        - **11th**: 11º ano
        - **12th**: 12º ano
        - **HS-grad**: Ensino secundário completo
        - **Some-college**: Algum ensino superior
        - **Assoc-voc**: Associado vocacional
        - **Assoc-acdm**: Associado académico
        - **Bachelors**: Licenciatura
        - **Masters**: Mestrado
        - **Prof-school**: Escola profissional
        - **Doctorate**: Doutoramento
        
        **💑 Marital-status (Estado Civil):**
        - **Married-civ-spouse**: Casado(a) - cônjuge civil
        - **Divorced**: Divorciado(a)
        - **Never-married**: Nunca casou
        - **Separated**: Separado(a)
        - **Widowed**: Viúvo(a)
        - **Married-spouse-absent**: Casado(a) - cônjuge ausente
        - **Married-AF-spouse**: Casado(a) - cônjuge nas forças armadas
        
        **💼 Occupation (Ocupação):**
        - **Tech-support**: Suporte técnico
        - **Craft-repair**: Artesanato/reparação
        - **Other-service**: Outros serviços
        - **Sales**: Vendas
        - **Exec-managerial**: Executivo/gestão
        - **Prof-specialty**: Especialista profissional
        - **Handlers-cleaners**: Manipuladores/limpeza
        - **Machine-op-inspct**: Operador de máquinas/inspetor
        - **Adm-clerical**: Administrativo/escritório
        - **Farming-fishing**: Agricultura/pesca
        - **Transport-moving**: Transporte/mudanças
        - **Priv-house-serv**: Serviços domésticos privados
        - **Protective-serv**: Serviços de proteção
        - **Armed-Forces**: Forças armadas
        
        **👥 Relationship (Relacionamento):**
        - **Wife**: Esposa
        - **Own-child**: Filho(a) próprio(a)
        - **Husband**: Marido
        - **Not-in-family**: Não está em família
        - **Other-relative**: Outro parente
        - **Unmarried**: Solteiro(a)
        
        **🌍 Race (Raça):**
        - **White**: Branco
        - **Asian-Pac-Islander**: Asiático-Ilhas do Pacífico
        - **Amer-Indian-Eskimo**: Índio americano-Esquimó
        - **Other**: Outro
        - **Black**: Negro
        
        **⚧ Sex (Sexo):**
        - **Male**: Masculino
        - **Female**: Feminino
        
        **🌏 Native-country (País de Origem):**
        - **United-States**: Estados Unidos
        - **Canada**: Canadá
        - **Germany**: Alemanha
        - **India**: Índia
        - **Japan**: Japão
        - **Other**: Outros países
        
        **🎯 Target (Variável Alvo):**
        - **<=50K**: Salário menor ou igual a 50.000 dólares/ano
        - **>50K**: Salário maior que 50.000 dólares/ano
        
        ---
        
        **💡 Dicas para Previsão:**
        - Educação e idade tendem a ser fatores importantes
        - Horas trabalhadas por semana influenciam o salário
        - Ocupações executivas/especializadas têm maior probabilidade de salário alto
        - Ganhos de capital podem indicar renda adicional
        """)

    # Verificar se o modelo e preprocessor existem
    if model is not None and preprocessor is not None:
        st.markdown("#### 📝 Entrada Manual de Dados")
        
        # Organizar campos em grupos lógicos com validação
        st.markdown("##### 👤 Informações Pessoais")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Validar idade com informações do tipo
            age_info = get_column_info()['age']
            age = st.number_input(
                f"🎂 Idade ({age_info['min']}-{age_info['max']} {age_info['unit']})", 
                min_value=age_info['min'], 
                max_value=age_info['max'], 
                value=30,
                step=1,
                help=f"{age_info['description']} - Range válido: {age_info['min']}-{age_info['max']} {age_info['unit']}"
            )
            
            # Mostrar opções traduzidas para sexo
            sex_options = ["Male", "Female"]
            translations = get_translations()
            sex_options_translated = [translate_value(opt, translations) for opt in sex_options]
            sex_translated = st.selectbox("⚧ Sexo", sex_options_translated, help="Sexo da pessoa")
            
            # Converter de volta para inglês
            reverse_translations = get_reverse_translations()
            sex = reverse_translations.get(sex_translated, sex_translated)
        
        with col2:
            # Raça com tradução
            race_options = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
            race_options_translated = [translate_value(opt, translations) for opt in race_options]
            race_translated = st.selectbox("🌍 Raça", race_options_translated, help="Origem étnica da pessoa")
            race = reverse_translations.get(race_translated, race_translated)
            
            # País com tradução
            country_options = ["United-States", "Canada", "Germany", "India", "Japan", "Mexico", "Other"]
            country_options_translated = [translate_value(opt, translations) for opt in country_options]
            native_country_translated = st.selectbox("🌏 País de Origem", country_options_translated, help="País de nascimento")
            native_country = reverse_translations.get(native_country_translated, native_country_translated)
        
        with col3:
            # Estado civil com tradução
            marital_options = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
            marital_options_translated = [translate_value(opt, translations) for opt in marital_options]
            marital_status_translated = st.selectbox("💑 Estado Civil", marital_options_translated, help="Estado civil atual")
            marital_status = reverse_translations.get(marital_status_translated, marital_status_translated)
            
            # Relacionamento com tradução
            relationship_options = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
            relationship_options_translated = [translate_value(opt, translations) for opt in relationship_options]
            relationship_translated = st.selectbox("👥 Relacionamento", relationship_options_translated, help="Relacionamento dentro da família")
            relationship = reverse_translations.get(relationship_translated, relationship_translated)
        
        st.markdown("##### 🎓 Educação e Trabalho")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Educação com tradução
            education_options = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
            education_options_translated = [translate_value(opt, translations) for opt in education_options]
            education_translated = st.selectbox("🎓 Nível de Educação", education_options_translated, help="Maior nível de educação completado")
            education = reverse_translations.get(education_translated, education_translated)
            
            # Anos de educação com validação
            edu_info = get_column_info()['education-num']
            education_num = st.number_input(
                f"📚 Anos de Educação ({edu_info['min']}-{edu_info['max']} {edu_info['unit']})",
                min_value=edu_info['min'],
                max_value=edu_info['max'],
                value=10,
                step=1,
                help=f"{edu_info['description']} - Range válido: {edu_info['min']}-{edu_info['max']} {edu_info['unit']}"
            )
        
        with col2:
            # Classe de trabalho com tradução
            workclass_options = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
            workclass_options_translated = [translate_value(opt, translations) for opt in workclass_options]
            workclass_translated = st.selectbox("🏢 Classe de Trabalho", workclass_options_translated, help="Tipo de empregador")
            workclass = reverse_translations.get(workclass_translated, workclass_translated)
            
            # Ocupação com tradução
            occupation_options = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
            occupation_options_translated = [translate_value(opt, translations) for opt in occupation_options]
            occupation_translated = st.selectbox("💼 Ocupação", occupation_options_translated, help="Ocupação profissional")
            occupation = reverse_translations.get(occupation_translated, occupation_translated)
        
        with col3:
            # Horas por semana com validação
            hours_info = get_column_info()['hours-per-week']
            hours_per_week = st.number_input(
                f"⏰ Horas por Semana ({hours_info['min']}-{hours_info['max']} {hours_info['unit']})",
                min_value=hours_info['min'],
                max_value=hours_info['max'],
                value=40,
                step=1,
                help=f"{hours_info['description']} - Range válido: {hours_info['min']}-{hours_info['max']} {hours_info['unit']}"
            )
        
        st.markdown("##### 💰 Informações Financeiras")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Ganho de capital com validação
            gain_info = get_column_info()['capital-gain']
            capital_gain = st.number_input(
                f"📈 Ganho de Capital (0-{gain_info['max']} {gain_info['unit']})",
                min_value=gain_info['min'],
                max_value=gain_info['max'],
                value=0,
                step=100,
                help=f"{gain_info['description']} - Range válido: {gain_info['min']}-{gain_info['max']} {gain_info['unit']}"
            )
        
        with col2:
            # Perda de capital com validação
            loss_info = get_column_info()['capital-loss']
            capital_loss = st.number_input(
                f"📉 Perda de Capital (0-{loss_info['max']} {loss_info['unit']})",
                min_value=loss_info['min'],
                max_value=loss_info['max'],
                value=0,
                step=50,
                help=f"{loss_info['description']} - Range válido: {loss_info['min']}-{loss_info['max']} {loss_info['unit']}"
            )
        
        with col3:
            # Peso final com validação
            fnlwgt_info = get_column_info()['fnlwgt']
            fnlwgt = st.number_input(
                f"⚖️ Peso Final (1-{fnlwgt_info['max']} {fnlwgt_info['unit']})",
                min_value=fnlwgt_info['min'],
                max_value=fnlwgt_info['max'],
                value=200000,
                step=10000,
                help=f"{fnlwgt_info['description']} - Range válido: {fnlwgt_info['min']}-{fnlwgt_info['max']} {fnlwgt_info['unit']}"
            )

        # Botão de previsão mais destacado
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            predict_button = st.button("🎯 **FAZER PREVISÃO**", type="primary", use_container_width=True)
        
        if predict_button:
            try:
                # Criar DataFrame com os dados inseridos
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
                
                # Preprocessar os dados
                input_processed = preprocessor.transform(input_data)
                
                # Fazer previsão
                prediction = model.predict(input_processed)[0]
                probability = model.predict_proba(input_processed)[0]
                
                # Mostrar resultado com mais destaque
                st.markdown("---")
                st.markdown("### 🎯 Resultado da Previsão")
                
                if prediction == 1:
                    st.success("### 🎉 **Salário > 50K** 💰")
                    st.info(f"**Probabilidade:** {probability[1]:.1%}")
                    
                    # Adicionar explicação
                    st.markdown("""
                    **Interpretação:** Com base nas características inseridas, o modelo prevê que esta pessoa 
                    tem alta probabilidade de ter um salário superior a 50.000 dólares anuais.
                    """)
                else:
                    st.warning("### 📊 **Salário ≤ 50K** 📉")
                    st.info(f"**Probabilidade:** {probability[0]:.1%}")
                    
                    # Adicionar explicação
                    st.markdown("""
                    **Interpretação:** Com base nas características inseridas, o modelo prevê que esta pessoa 
                    tem alta probabilidade de ter um salário menor ou igual a 50.000 dólares anuais.
                    """)
                
                # Mostrar dados de entrada organizados
                with st.expander("📋 Ver Resumo dos Dados Inseridos"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Informações Pessoais:**")
                        st.write(f"• Idade: {age} anos")
                        st.write(f"• Sexo: {sex}")
                        st.write(f"• Raça: {race}")
                        st.write(f"• País: {native_country}")
                        st.write(f"• Estado Civil: {marital_status}")
                        st.write(f"• Relacionamento: {relationship}")
                        
                        st.markdown("**Informações Financeiras:**")
                        st.write(f"• Ganho Capital: ${capital_gain:,}")
                        st.write(f"• Perda Capital: ${capital_loss:,}")
                        st.write(f"• Peso Final: {fnlwgt:,}")
                    
                    with col2:
                        st.markdown("**Educação e Trabalho:**")
                        st.write(f"• Educação: {education}")
                        st.write(f"• Anos Educação: {education_num}")
                        st.write(f"• Classe Trabalho: {workclass}")
                        st.write(f"• Ocupação: {occupation}")
                        st.write(f"• Horas/Semana: {hours_per_week}")
                    
                    # Mostrar DataFrame completo
                    st.markdown("**Dados Completos:**")
                    st.dataframe(input_data, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Erro na previsão: {str(e)}")
                st.info("💡 Verifique se todos os campos estão preenchidos corretamente.")

        # Upload de arquivo (apenas para utilizadores com permissão)
        if user_permissions["upload_files"]:
            st.markdown("---")
            st.markdown("#### 📁 Upload de Arquivo CSV")
            st.info("💡 Carregue um arquivo CSV com as mesmas colunas descritas acima para fazer previsões em lote.")
            
            uploaded_file = st.file_uploader("Carregar ficheiro CSV para previsão", type=["csv"],
                                           help="O arquivo deve conter as colunas: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country")
            
            if uploaded_file is not None:
                try:
                    new_data = pd.read_csv(uploaded_file)
                    st.write("📋 Pré-visualização dos dados carregados:")
                    st.dataframe(new_data.head(), use_container_width=True)
                    
                    st.info(f"📊 Total de registos carregados: **{len(new_data)}**")
                    
                    if st.button("🎯 Fazer Previsões em Lote", type="primary"):
                        try:
                            # Preprocessar os dados
                            new_data_processed = preprocessor.transform(new_data)
                            
                            # Fazer previsões
                            predictions = model.predict(new_data_processed)
                            probabilities = model.predict_proba(new_data_processed)
                            
                            # Adicionar resultados ao DataFrame
                            results = new_data.copy()
                            results['Previsão'] = ['> 50K' if p == 1 else '≤ 50K' for p in predictions]
                            results['Probabilidade_Alto'] = probabilities[:, 1]
                            results['Probabilidade_Baixo'] = probabilities[:, 0]
                            
                            st.success("✅ Previsões realizadas com sucesso!")
                            
                            # Estatísticas dos resultados
                            high_salary_count = (predictions == 1).sum()
                            low_salary_count = (predictions == 0).sum()
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Total Previsões", len(predictions))
                            col2.metric("Salário > 50K", high_salary_count)
                            col3.metric("Salário ≤ 50K", low_salary_count)
                            
                            st.dataframe(results, use_container_width=True)
                            
                            # Opção para download
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="📥 Baixar Resultados (CSV)",
                                data=csv,
                                file_name="previsoes_salario.csv",
                                mime="text/csv",
                                help="Clique para baixar os resultados em formato CSV"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Erro nas previsões: {str(e)}")
                            st.info("💡 Verifique se o arquivo CSV tem as colunas corretas.")
                    
                except Exception as e:
                    st.error(f"❌ Erro ao carregar arquivo: {str(e)}")
                    st.info("💡 Certifique-se de que o arquivo é um CSV válido.")

    else:
        st.error("❌ Modelo ou preprocessor não encontrados!")
        if user_permissions["view_system_info"]:
            st.markdown("""
            **Para resolver este problema:**
            
            1. Execute primeiro o script principal: `python projeto_salario.py`
            2. Aguarde o treinamento dos modelos
            3. Verifique se os seguintes arquivos foram criados:
               - `random_forest_model.joblib`
               - `preprocessor.joblib`
               - `feature_info.joblib`
            4. Execute novamente o dashboard: `streamlit run dashboard_app.py`
            """)
        else:
            st.info("💡 Contacte o administrador para resolver este problema.")
else:
    st.info("🔒 Não tem permissão para aceder à funcionalidade de previsões.")

# ======================================
# 8. INFORMAÇÕES DO SISTEMA
# ======================================
if user_permissions["view_system_info"]:
    with st.expander("ℹ️ Informações do Sistema"):
        shap_status = "✅" if SHAP_AVAILABLE else "⚠️"
        shap_text = "Disponível" if SHAP_AVAILABLE else "Não instalado"
        model_status = "✅" if model is not None else "❌"
        model_text = "Carregado" if model is not None else "Não encontrado"
        
        st.markdown(f"""
        **Utilizador Atual:** `{st.session_state['current_user']}`
        
        **Permissões:**
        - 📊 Ver dados: {'✅' if user_permissions['view_data'] else '❌'}
        - 📈 Ver visualizações: {'✅' if user_permissions['view_visualizations'] else '❌'}
        - 🤖 Ver modelos: {'✅' if user_permissions['view_models'] else '❌'}
        - 🔮 Fazer previsões: {'✅' if user_permissions['make_predictions'] else '❌'}
        - 📁 Upload ficheiros: {'✅' if user_permissions['upload_files'] else '❌'}
        - ⚙️ Info sistema: {'✅' if user_permissions['view_system_info'] else '❌'}
        
        **Funcionalidades do Dashboard:**
        - 📊 Visualizar distribuições dos dados
        - 🔍 Analisar importância das features
        - 🎯 Explorar clustering dos dados
        - 🔮 Fazer previsões individuais e em lote
        
        **Arquivos Necessários:**
        - Dados: `4-Carateristicas_salario.csv`
        - Modelo: `random_forest_model.joblib`
        - Preprocessor: `preprocessor.joblib`
        - Features: `feature_info.joblib`
        - Gráficos: pasta `imagens/`
        
        **Status do Sistema:**
        - ✅ Streamlit: Funcionando
        - {shap_status} SHAP: {shap_text}
        - {model_status} Modelo: {model_text}
        - 📊 Dados: {'✅ Carregados' if not df.empty else '❌ Não encontrados'}
        - 📈 Gráficos: {'✅' if os.path.exists('imagens') else '❌'} {'Disponíveis' if os.path.exists('imagens') else 'Pasta não encontrada'}
        """)

# Adicionar função helper para verificar status das imagens
def get_image_status():
    """Verificar status de todas as imagens geradas"""
    expected_images = {
        # Distribuições
        "hist_age.png": "Histograma da Idade",
        "hist_fnlwgt.png": "Histograma do Peso Final", 
        "hist_education-num.png": "Histograma dos Anos de Educação",
        "hist_capital-gain.png": "Histograma do Ganho de Capital",
        "hist_capital-loss.png": "Histograma da Perda de Capital",
        "hist_hours-per-week.png": "Histograma das Horas por Semana",
        
        # Análises categóricas
        "workclass_distribution.png": "Distribuição da Classe de Trabalho",
        "education_distribution.png": "Distribuição da Educação",
        "marital_status_distribution.png": "Distribuição do Estado Civil",
        "occupation_distribution.png": "Distribuição das Ocupações",
        "salary_distribution.png": "Distribuição dos Salários",
        
        # Correlações
        "correlacao.png": "Matriz de Correlação",
        
        # Importância das features
        "feature_importance_rf.png": "Importância das Features - Random Forest",
        "coefficients_lr.png": "Coeficientes - Regressão Logística",
        
        # Clustering
        "kmeans_clusters.png": "Clusters K-Means",
        "pca_analysis.png": "Análise PCA",
        
        # SHAP (se disponível)
        "shap_summary.png": "SHAP Summary Plot",
        "shap_waterfall.png": "SHAP Waterfall Plot", 
        "shap_dependence.png": "SHAP Dependence Plot",
        
        # Outros gráficos
        "salary_by_education.png": "Salário por Educação",
        "salary_by_age.png": "Salário por Idade",
        "confusion_matrix.png": "Matriz de Confusão"
    }
    
    status = {}
    for filename, description in expected_images.items():
        path = f"imagens/{filename}"
        exists = os.path.exists(path)
        size = 0
        if exists:
            try:
                size = os.path.getsize(path)
            except:
                size = 0
        
        status[filename] = {
            "exists": exists,
            "size": size,
            "description": description,
            "path": path
        }
    
    return status

# Adicionar na seção de informações do sistema
if user_permissions["view_system_info"]:
    with st.expander("📈 Status dos Gráficos Gerados"):
        image_status = get_image_status()
        
        st.markdown("#### 📊 Gráficos Disponíveis")
        
        # Categorizar imagens
        categories = {
            "📊 Distribuições": [f for f in image_status.keys() if f.startswith("hist_")],
            "📈 Análises Categóricas": [f for f in image_status.keys() if f.endswith("_distribution.png")],
            "🔗 Correlações": ["correlacao.png"],
            "🎯 Importância": ["feature_importance_rf.png", "coefficients_lr.png"],
            "🧩 Clustering": ["kmeans_clusters.png", "pca_analysis.png"],
            "🔍 SHAP": [f for f in image_status.keys() if f.startswith("shap_")],
            "📋 Outros": [f for f in image_status.keys() if f not in sum([
                [f for f in image_status.keys() if f.startswith("hist_")],
                [f for f in image_status.keys() if f.endswith("_distribution.png")],
                ["correlacao.png"],
                ["feature_importance_rf.png", "coefficients_lr.png"],
                ["kmeans_clusters.png", "pca_analysis.png"],
                [f for f in image_status.keys() if f.startswith("shap_")]
            ], [])]
        }
        
        for category, files in categories.items():
            if files:  # Só mostrar categorias que têm arquivos
                st.markdown(f"**{category}**")
                cols = st.columns(3)
                
                for idx, filename in enumerate(files):
                    if filename in image_status:
                        status = image_status[filename]
                        with cols[idx % 3]:
                            if status["exists"] and status["size"] > 0:
                                st.success(f"✅ {status['description']}")
                                st.caption(f"📦 {status['size'] // 1024} KB")
                            else:
                                st.warning(f"⚠️ {status['description']}")
                                st.caption("Arquivo não encontrado")
                
                st.markdown("---")
        
        # Estatísticas gerais
        total_images = len(image_status)
        existing_images = sum(1 for s in image_status.values() if s["exists"] and s["size"] > 0)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Gráficos", total_images)
        col2.metric("Gráficos Gerados", existing_images)
        col3.metric("Taxa de Conclusão", f"{(existing_images/total_images)*100:.1f}%")

def safe_image_display(image_path, caption, container=None):
    """Exibir imagem com compatibilidade entre versões do Streamlit"""
    try:
        # Tentar primeiro use_container_width (versões mais recentes)
        if container:
            container.image(image_path, caption=caption, use_container_width=True)
        else:
            st.image(image_path, caption=caption, use_container_width=True)
    except TypeError:
        # Fallback para use_column_width (versões mais antigas)
        if container:
            container.image(image_path, caption=caption, use_column_width=True)
        else:
            st.image(image_path, caption=caption, use_column_width=True)

def safe_button(label, button_type="secondary", container=None):
    """Botão com compatibilidade entre versões"""
    try:
        if container:
            return container.button(label, type=button_type, use_container_width=True)
        else:
            return st.button(label, type=button_type, use_container_width=True)
    except TypeError:
        if container:
            return container.button(label, type=button_type, use_column_width=True)
        else:
            return st.button(label, type=button_type, use_column_width=True)

def safe_dataframe(df, container=None):
    """DataFrame com compatibilidade"""
    try:
        if container:
            container.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
    except TypeError:
        if container:
            container.dataframe(df, use_column_width=True)
        else:
            st.dataframe(df, use_column_width=True)
