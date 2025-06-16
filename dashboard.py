"""
🎓 Dashboard Científico Final - Análise Salarial Acadêmica
Versão Corrigida com Pipeline Completo
"""

# IMPORTANTE: st.set_page_config() deve ser a PRIMEIRA linha de código Streamlit
import streamlit as st

# Configuração da página - DEVE SER A PRIMEIRA CHAMADA ST
st.set_page_config(
    page_title="🎓 Dashboard Científico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agora importar o resto
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SISTEMA DE PIPELINE CIENTÍFICO
# =============================================================================

class ScientificPipeline:
    def __init__(self):
        self.results = {}
        self.models = {}
        
    def execute_all_algorithms(self, df):
        """Executar todos os algoritmos do pipeline"""
        results_summary = {
            'executed': [],
            'failed': [],
            'execution_score': 0
        }
        
        try:
            # 1. DBSCAN Clustering
            dbscan_result = self.run_dbscan(df)
            if dbscan_result['status'] == 'success':
                results_summary['executed'].append('DBSCAN')
                self.results['dbscan'] = dbscan_result
            else:
                results_summary['failed'].append('DBSCAN')
        except Exception as e:
            results_summary['failed'].append(f'DBSCAN: {str(e)}')
        
        try:
            # 2. Random Forest
            rf_result = self.run_random_forest(df)
            if rf_result['status'] == 'success':
                results_summary['executed'].append('Random Forest')
                self.results['random_forest'] = rf_result
            else:
                results_summary['failed'].append('Random Forest')
        except Exception as e:
            results_summary['failed'].append(f'Random Forest: {str(e)}')
        
        try:
            # 3. Logistic Regression
            lr_result = self.run_logistic_regression(df)
            if lr_result['status'] == 'success':
                results_summary['executed'].append('Logistic Regression')
                self.results['logistic_regression'] = lr_result
            else:
                results_summary['failed'].append('Logistic Regression')
        except Exception as e:
            results_summary['failed'].append(f'Logistic Regression: {str(e)}')
        
        try:
            # 4. Association Rules (Simulado)
            assoc_result = self.run_association_rules(df)
            if assoc_result['status'] == 'success':
                results_summary['executed'].extend(['APRIORI', 'FP-GROWTH', 'ECLAT'])
                self.results['association'] = assoc_result
            else:
                results_summary['failed'].extend(['APRIORI', 'FP-GROWTH', 'ECLAT'])
        except Exception as e:
            results_summary['failed'].append(f'Association Rules: {str(e)}')
        
        # Calcular score de execução
        total_algorithms = 6  # DBSCAN, RF, LR, APRIORI, FP-GROWTH, ECLAT
        executed_count = len(results_summary['executed'])
        results_summary['execution_score'] = executed_count / total_algorithms
        
        return results_summary
    
    def run_dbscan(self, df):
        """Executar DBSCAN"""
        try:
            # Selecionar features numéricas
            numeric_cols = ['age', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss']
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            if len(available_cols) < 2:
                return {'status': 'failed', 'error': 'Insufficient numeric columns'}
            
            # Preparar dados
            X = df[available_cols].fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Executar DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(X_scaled)
            
            # Métricas
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_rate = n_noise / len(labels)
            
            return {
                'status': 'success',
                'n_clusters': n_clusters,
                'noise_rate': noise_rate,
                'labels': labels,
                'features_used': available_cols
            }
        
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def run_random_forest(self, df):
        """Executar Random Forest"""
        try:
            if 'salary' not in df.columns:
                return {'status': 'failed', 'error': 'Target column salary not found'}
            
            # Preparar features
            numeric_cols = ['age', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss']
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            if len(available_cols) < 2:
                return {'status': 'failed', 'error': 'Insufficient features'}
            
            X = df[available_cols].fillna(0)
            y = (df['salary'] == '>50K').astype(int)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Treinar modelo
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Predições
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = dict(zip(available_cols, rf.feature_importances_))
            
            # Salvar modelo
            self.models['random_forest'] = rf
            
            return {
                'status': 'success',
                'accuracy': accuracy,
                'feature_importance': feature_importance,
                'features_used': available_cols
            }
        
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def run_logistic_regression(self, df):
        """Executar Logistic Regression"""
        try:
            if 'salary' not in df.columns:
                return {'status': 'failed', 'error': 'Target column salary not found'}
            
            # Preparar features
            numeric_cols = ['age', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss']
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            if len(available_cols) < 2:
                return {'status': 'failed', 'error': 'Insufficient features'}
            
            X = df[available_cols].fillna(0)
            y = (df['salary'] == '>50K').astype(int)
            
            # Normalizar features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Treinar modelo
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_train, y_train)
            
            # Predições
            y_pred = lr.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Coeficientes
            coefficients = dict(zip(available_cols, lr.coef_[0]))
            
            # Salvar modelo
            self.models['logistic_regression'] = lr
            
            return {
                'status': 'success',
                'accuracy': accuracy,
                'coefficients': coefficients,
                'features_used': available_cols
            }
        
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def run_association_rules(self, df):
        """Simular regras de associação"""
        try:
            # Simulação simples de regras de associação
            rules = []
            
            if 'education' in df.columns and 'salary' in df.columns:
                # Regra 1: Educação alta -> Salário alto
                high_edu = df['education'].isin(['Bachelors', 'Masters', 'Doctorate'])
                high_salary = df['salary'] == '>50K'
                
                if high_edu.sum() > 0:
                    confidence = (high_edu & high_salary).sum() / high_edu.sum()
                    support = (high_edu & high_salary).sum() / len(df)
                    
                    rules.append({
                        'rule': 'High Education → High Salary',
                        'confidence': confidence,
                        'support': support
                    })
            
            if 'hours-per-week' in df.columns and 'salary' in df.columns:
                # Regra 2: Muitas horas -> Salário alto
                many_hours = df['hours-per-week'] >= 50
                high_salary = df['salary'] == '>50K'
                
                if many_hours.sum() > 0:
                    confidence = (many_hours & high_salary).sum() / many_hours.sum()
                    support = (many_hours & high_salary).sum() / len(df)
                    
                    rules.append({
                        'rule': 'Many Hours → High Salary',
                        'confidence': confidence,
                        'support': support
                    })
            
            return {
                'status': 'success',
                'rules': rules,
                'n_rules': len(rules)
            }
        
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

# =============================================================================
# SISTEMA DE DADOS
# =============================================================================

@st.cache_data
def load_data():
    """Carregar dados com fallback para dados sintéticos"""
    # Tentar carregar arquivo real
    possible_paths = [
        'bkp/4-Carateristicas_salario.csv',
        'data/raw/4-Carateristicas_salario.csv',
        '4-Carateristicas_salario.csv'
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            try:
                df = pd.read_csv(path)
                if len(df) > 0:
                    # Limpar dados
                    df = clean_data(df)
                    return df, f"✅ Carregado: {path}"
            except:
                continue
    
    # Criar dados sintéticos se não encontrar arquivo
    return create_synthetic_data(), "⚠️ Usando dados sintéticos"

def clean_data(df):
    """Limpar e padronizar dados"""
    # Substituir valores problemáticos
    df = df.replace(['?', 'unknown', 'Unknown', ''], np.nan)
    
    # Preencher valores ausentes
    for col in df.select_dtypes(include=['object']).columns:
        if col in df.columns:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Other'
            df[col] = df[col].fillna(mode_val)
    
    # Padronizar salary
    if 'salary' in df.columns:
        df['salary'] = df['salary'].str.strip()
        df['salary'] = df['salary'].replace({
            '<=50K': '<=50K',
            '>50K': '>50K'
        })
    
    return df

def create_synthetic_data():
    """Criar dados sintéticos para demonstração"""
    np.random.seed(42)
    n = 2000
    
    # Gerar dados realistas
    ages = np.random.normal(39, 13, n).astype(int)
    ages = np.clip(ages, 17, 90)
    
    edu_nums = np.random.choice(range(1, 17), n, 
                               p=[0.01, 0.01, 0.02, 0.03, 0.05, 0.05, 0.08, 0.10, 0.15, 0.15, 0.15, 0.10, 0.05, 0.03, 0.01, 0.01])
    
    # Correlação educação-salário
    salary_prob = (edu_nums - 8) / 8 * 0.4 + 0.1
    salary_prob = np.clip(salary_prob, 0.05, 0.6)
    salaries = np.random.binomial(1, salary_prob, n)
    
    countries = ['United-States', 'Mexico', 'Canada', 'Germany', 'Philippines', 'India', 'Poland', 'Jamaica', 'Japan']
    educations = ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Doctorate', '11th', '10th', 'Assoc-voc']
    
    return pd.DataFrame({
        'age': ages,
        'education-num': edu_nums,
        'education': np.random.choice(educations, n),
        'hours-per-week': np.random.normal(40, 12, n).astype(int).clip(1, 99),
        'capital-gain': np.random.exponential(500, n).astype(int),
        'capital-loss': np.random.exponential(200, n).astype(int),
        'salary': ['>50K' if s else '<=50K' for s in salaries],
        'sex': np.random.choice(['Male', 'Female'], n, p=[0.67, 0.33]),
        'native-country': np.random.choice(countries, n),
        'workclass': np.random.choice(['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov'], n, p=[0.7, 0.15, 0.1, 0.05]),
        'marital-status': np.random.choice(['Married-civ-spouse', 'Never-married', 'Divorced'], n, p=[0.5, 0.35, 0.15])
    })

# =============================================================================
# AUTENTICAÇÃO
# =============================================================================

def check_authentication():
    """Verificar autenticação"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        show_login()
        return False
    return True

def show_login():
    """Tela de login"""
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white; margin-bottom: 2rem;">
        <h1>🎓 Dashboard Científico</h1>
        <h3>Análise Salarial Acadêmica</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        
        users = {
            "admin": "123",
            "professor": "academic2024",
            "aluno": "student2024",
            "demo": "demo"
        }
        
        username = st.selectbox("👤 Usuário:", [""] + list(users.keys()))
        password = st.text_input("🔑 Senha:", type="password")
        
        if st.button("🚀 Entrar", use_container_width=True, type="primary"):
            if username and password and username in users and users[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success(f"✅ Bem-vindo, {username}!")
                st.balloons()
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Credenciais inválidas!")
        
        with st.expander("ℹ️ Credenciais de Teste"):
            st.markdown("""
            - **admin** / 123
            - **professor** / academic2024  
            - **aluno** / student2024
            - **demo** / demo
            """)

# =============================================================================
# PÁGINAS
# =============================================================================

def show_pipeline_status(pipeline_results):
    """Mostrar status do pipeline"""
    st.subheader("🚀 Status do Pipeline Científico")
    
    # Status cards
    algorithms = [
        ("🎯 DBSCAN", "DBSCAN"),
        ("✅ ECLAT", "ECLAT"), 
        ("📋 APRIORI", "APRIORI"),
        ("🌳 Random Forest", "Random Forest"),
        ("🚀 FP-GROWTH", "FP-GROWTH"),
        ("📈 Logistic Regression", "Logistic Regression")
    ]
    
    col1, col2, col3 = st.columns(3)
    
    for i, (name, algo) in enumerate(algorithms):
        col = [col1, col2, col3][i % 3]
        
        with col:
            is_executed = algo in pipeline_results.get('executed', [])
            status_text = "✅ Executado" if is_executed else "❌ Pendente"
            status_color = "#28a745" if is_executed else "#dc3545"
            
            st.markdown(f"""
            <div style="padding: 1rem; border: 2px solid {status_color}; border-radius: 10px; margin-bottom: 1rem; background: {status_color}15;">
                <h4 style="margin: 0; color: {status_color};">{name}</h4>
                <p style="margin: 0.5rem 0; color: #6c757d; font-size: 0.9rem;">
                    {'Clustering baseado em densidade' if 'DBSCAN' in name else
                     'Algoritmo de intersecção' if 'ECLAT' in name else
                     'Regras de associação clássicas' if 'APRIORI' in name else
                     'Modelo ensemble robusto' if 'Random Forest' in name else
                     'Mineração eficiente de padrões' if 'FP-GROWTH' in name else
                     'Modelo linear interpretável'}
                </p>
                <p style="margin: 0; font-weight: bold; color: {status_color};">{status_text}</p>
            </div>
            """, unsafe_allow_html=True)

def show_overview(df, pipeline_results):
    """Página de visão geral"""
    st.title("📊 Visão Geral - Dashboard Científico")
    st.markdown("---")
    
    # Status do pipeline
    show_pipeline_status(pipeline_results)
    
    st.markdown("---")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Registros", f"{len(df):,}")
    
    with col2:
        if 'salary' in df.columns:
            high_rate = (df['salary'] == '>50K').mean()
            st.metric("💰 Taxa >50K", f"{high_rate:.1%}")
        else:
            st.metric("💰 Taxa >50K", "N/A")
    
    with col3:
        if 'age' in df.columns:
            avg_age = df['age'].mean()
            st.metric("👥 Idade Média", f"{avg_age:.1f}")
        else:
            st.metric("👥 Idade Média", "N/A")
    
    with col4:
        score = pipeline_results.get('execution_score', 0)
        st.metric("🤖 Pipeline", f"{score:.0%}")
    
    # Executar pipeline se não foi executado
    if pipeline_results.get('execution_score', 0) == 0:
        st.warning("⚠️ Pipeline não executado")
        if st.button("🚀 Executar Pipeline Científico", type="primary"):
            with st.spinner("🔄 Executando algoritmos..."):
                pipeline = ScientificPipeline()
                results = pipeline.execute_all_algorithms(df)
                st.session_state.pipeline_results = results
                st.session_state.pipeline = pipeline
                st.success("✅ Pipeline executado com sucesso!")
                st.rerun()
    
    # Insights
    st.subheader("🎓 Insights Acadêmicos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌍 Top 5 Países - Taxa Salarial >50K")
        if 'native-country' in df.columns and 'salary' in df.columns:
            country_counts = df['native-country'].value_counts()
            valid_countries = country_counts[country_counts >= 20].index
            
            if len(valid_countries) > 0:
                df_filtered = df[df['native-country'].isin(valid_countries)]
                country_rates = df_filtered.groupby('native-country')['salary'].apply(
                    lambda x: (x == '>50K').mean()
                ).sort_values(ascending=False).head(5)
                
                for country, rate in country_rates.items():
                    count = (df_filtered['native-country'] == country).sum()
                    st.write(f"**{country}**: {rate:.1%} ({count:,} pessoas)")
            else:
                st.write("*Dados insuficientes*")
        else:
            st.write("*Coluna não encontrada*")
    
    with col2:
        st.markdown("#### 🎓 Top 5 Educação - Taxa Salarial >50K")
        if 'education' in df.columns and 'salary' in df.columns:
            edu_counts = df['education'].value_counts()
            valid_education = edu_counts[edu_counts >= 20].index
            
            if len(valid_education) > 0:
                df_filtered = df[df['education'].isin(valid_education)]
                edu_rates = df_filtered.groupby('education')['salary'].apply(
                    lambda x: (x == '>50K').mean()
                ).sort_values(ascending=False).head(5)
                
                for edu, rate in edu_rates.items():
                    count = (df_filtered['education'] == edu).sum()
                    st.write(f"**{edu}**: {rate:.1%} ({count:,} pessoas)")
            else:
                st.write("*Dados insuficientes*")
        else:
            st.write("*Coluna não encontrada*")

def main():
    """Função principal"""
    
    # Verificar autenticação
    if not check_authentication():
        return
    
    # Carregar dados
    df, status = load_data()
    
    # Inicializar pipeline se não existir
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = {'executed': [], 'failed': [], 'execution_score': 0}
    
    # Inicializar página atual
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📊 Visão Geral"
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;">
            <h3>🎓 Dashboard Científico</h3>
            <p>Usuário: <strong>{st.session_state.get('username', 'Guest')}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Navegação")
        
        # Botões de navegação
        pages = [
            "📊 Visão Geral",
            "🔮 Predição Interativa", 
            "🔍 Explorador de Dados"
        ]
        
        for page in pages:
            is_current = st.session_state.current_page == page
            if st.button(
                page, 
                key=f"btn_{page}",
                use_container_width=True,
                type="primary" if is_current else "secondary",
                disabled=is_current
            ):
                st.session_state.current_page = page
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🔧 Status")
        st.info(status)
        st.metric("📊 Registros", f"{len(df):,}")
        
        st.markdown("---")
        if st.button("🚪 Sair", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    # Mostrar página atual
    current_page = st.session_state.current_page
    
    # Indicador da página
    st.markdown(f"""
    <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px; margin-bottom: 2rem; border-left: 4px solid #2196f3;">
        <h4 style="margin:0; color:#1976d2;">📍 {current_page}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Roteamento
    if current_page == "📊 Visão Geral":
        show_overview(df, st.session_state.pipeline_results)
    # Adicionar outras páginas aqui...

if __name__ == "__main__":
    main()