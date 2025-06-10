"""
🌍 Sistema de Internacionalização (i18n)
Gerenciamento completo de traduções JSON
"""

import streamlit as st
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class I18nSystem:
    """Sistema de internacionalização com JSON"""
    
    def __init__(self):
        self.translations_dir = Path("translate")
        self.translations = {}
        self.config = self._load_config()
        self.default_language = self.config.get('default_language', 'pt')
        
        # Carregar todas as traduções
        self._load_all_translations()
        
        # Inicializar idioma
        if 'language' not in st.session_state:
            st.session_state.language = self.default_language
    
    def _load_config(self):
        """Carregar configuração de idiomas"""
        config_file = self.translations_dir / "config.json"
        try:
            if config_file.exists():
                return json.loads(config_file.read_text(encoding='utf-8'))
        except Exception as e:
            logger.error(f"Erro ao carregar config i18n: {e}")
        
        # Configuração padrão
        return {
            "default_language": "pt",
            "available_languages": {
                "pt": {"name": "🇵🇹 Português", "flag": "🇵🇹"},
                "en": {"name": "🇬🇧 English", "flag": "🇬🇧"}
            }
        }
    
    def _load_all_translations(self):
        """Carregar todas as traduções disponíveis"""
        if not self.translations_dir.exists():
            self._create_default_translations()
            return
        
        for lang_file in self.translations_dir.glob("*.json"):
            if lang_file.stem == "config":
                continue
                
            try:
                lang_code = lang_file.stem
                self.translations[lang_code] = json.loads(
                    lang_file.read_text(encoding='utf-8')
                )
                logger.info(f"✅ Tradução carregada: {lang_code}")
            except Exception as e:
                logger.error(f"Erro ao carregar {lang_file}: {e}")
    
    def _create_default_translations(self):
        """Criar traduções padrão se não existirem"""
        try:
            self.translations_dir.mkdir(exist_ok=True)
            
            # Traduções mínimas em português
            default_pt = {
                "app": {"title": "💰 Dashboard de Análise Salarial"},
                "auth": {
                    "login_title": "🔓 Acesso ao Sistema",
                    "username": "👤 Utilizador", 
                    "password": "🔑 Palavra-passe",
                    "login_button": "🚀 Entrar",
                    "demo_button": "🎮 Demo",
                    "logout": "🚪 Sair",
                    "welcome": "Bem-vindo",
                    "invalid_credentials": "❌ Credenciais inválidas!",
                    "logout_success": "✅ Logout realizado com sucesso!",
                    "logged_user": "👤 Utilizador Ligado",
                    "role": "🎯 Papel"
                },
                "navigation": {
                    "title": "Navegação",
                    "overview": "📊 Visão Geral",
                    "exploratory": "📈 Análise Exploratória", 
                    "models": "🤖 Modelos ML",
                    "prediction": "🔮 Predição"
                },
                "data": {
                    "records": "📋 Registos",
                    "columns": "📊 Colunas",
                    "high_salary": "💰 Salário Alto",
                    "missing": "❌ Em Falta",
                    "age": "🎂 Idade",
                    "salary": "💰 Salário",
                    "education": "🎓 Educação",
                    "sex": "👥 Sexo",
                    "workclass": "💼 Classe Trabalhadora",
                    "marital_status": "💑 Estado Civil"
                },
                "charts": {
                    "salary_distribution": "💰 Distribuição de Salários",
                    "sex_distribution": "👥 Distribuição por Sexo",
                    "age_distribution": "📊 Distribuição de Idades",
                    "correlation_matrix": "🔗 Matriz de Correlação"
                },
                "messages": {
                    "pipeline_needed": "⚠️ Execute: python main.py",
                    "success": "✅ Sucesso!",
                    "error": "❌ Erro"
                },
                "models": {
                    "random_forest": "🌲 Random Forest",
                    "logistic_regression": "📊 Regressão Logística",
                    "svm": "🎯 SVM",
                    "accuracy": "📊 Precisão",
                    "precision": "🎯 Precision"
                }
            }
            
            # Traduções em inglês
            default_en = {
                "app": {"title": "💰 Salary Analysis Dashboard"},
                "auth": {
                    "login_title": "🔓 System Access",
                    "username": "👤 Username",
                    "password": "🔑 Password", 
                    "login_button": "🚀 Login",
                    "demo_button": "🎮 Demo",
                    "logout": "🚪 Logout",
                    "welcome": "Welcome",
                    "invalid_credentials": "❌ Invalid credentials!",
                    "logout_success": "✅ Logout successful!",
                    "logged_user": "👤 Logged User",
                    "role": "🎯 Role"
                },
                "navigation": {
                    "title": "Navigation",
                    "overview": "📊 Overview",
                    "exploratory": "📈 Exploratory Analysis",
                    "models": "🤖 ML Models", 
                    "prediction": "🔮 Prediction"
                },
                "data": {
                    "records": "📋 Records",
                    "columns": "📊 Columns",
                    "high_salary": "💰 High Salary",
                    "missing": "❌ Missing",
                    "age": "🎂 Age",
                    "salary": "💰 Salary",
                    "education": "🎓 Education",
                    "sex": "👥 Sex",
                    "workclass": "💼 Work Class",
                    "marital_status": "💑 Marital Status"
                },
                "charts": {
                    "salary_distribution": "💰 Salary Distribution",
                    "sex_distribution": "👥 Distribution by Sex",
                    "age_distribution": "📊 Age Distribution",
                    "correlation_matrix": "🔗 Correlation Matrix"
                },
                "messages": {
                    "pipeline_needed": "⚠️ Execute: python main.py",
                    "success": "✅ Success!",
                    "error": "❌ Error"
                },
                "models": {
                    "random_forest": "🌲 Random Forest",
                    "logistic_regression": "📊 Logistic Regression",
                    "svm": "🎯 SVM",
                    "accuracy": "📊 Accuracy",
                    "precision": "🎯 Precision"
                }
            }
            
            # Salvar arquivos
            (self.translations_dir / "pt.json").write_text(
                json.dumps(default_pt, indent=2, ensure_ascii=False), encoding='utf-8'
            )
            (self.translations_dir / "en.json").write_text(
                json.dumps(default_en, indent=2, ensure_ascii=False), encoding='utf-8'
            )
            
            # Config
            default_config = {
                "default_language": "pt",
                "available_languages": {
                    "pt": {"name": "🇵🇹 Português", "flag": "🇵🇹"},
                    "en": {"name": "🇬🇧 English", "flag": "🇬🇧"}
                }
            }
            (self.translations_dir / "config.json").write_text(
                json.dumps(default_config, indent=2), encoding='utf-8'
            )
            
            # Recarregar
            self._load_all_translations()
            
            st.success("✅ Estrutura de tradução criada automaticamente!")
            
        except Exception as e:
            st.error(f"❌ Erro ao criar estrutura: {e}")
    
    def get_language(self):
        """Obter idioma atual da sessão"""
        return st.session_state.get('language', self.default_language)
    
    def set_language(self, lang_code):
        """Definir idioma atual"""
        available_langs = self.get_available_languages()
        if lang_code in available_langs:
            st.session_state.language = lang_code
            return True
        return False
    
    def get_available_languages(self):
        """Obter idiomas disponíveis"""
        langs = self.config.get('available_languages', {})
        # Garantir que não está vazio
        if not langs:
            return {
                "pt": {"name": "🇵🇹 Português", "flag": "🇵🇹"},
                "en": {"name": "🇬🇧 English", "flag": "🇬🇧"}
            }
        return langs
    
    def t(self, key, fallback=None):
        """
        Traduzir chave usando notação de ponto
        Exemplo: t('auth.login_button') -> busca translations[lang]['auth']['login_button']
        """
        current_lang = self.get_language()
        
        # Buscar tradução no idioma atual
        text = self._get_nested_value(self.translations.get(current_lang, {}), key)
        
        # Fallback para inglês se não encontrar
        if text is None and current_lang != 'en':
            text = self._get_nested_value(self.translations.get('en', {}), key)
        
        # Fallback final
        if text is None:
            text = fallback if fallback is not None else key
        
        return text
    
    def _get_nested_value(self, data, key):
        """Buscar valor usando notação de ponto (ex: 'auth.login_button')"""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current
    
    def show_language_selector(self):
        """Mostrar seletor de idioma na sidebar"""
        with st.sidebar:
            st.markdown("---")
            
            current_lang = self.get_language()
            available_langs = self.get_available_languages()
            
            # PROTEÇÃO: Verificar se há idiomas disponíveis
            if not available_langs:
                st.warning("⚠️ Nenhum idioma configurado")
                return
            
            # Criar lista de opções
            lang_options = []
            lang_codes = []
            current_index = 0
            
            try:
                for i, (code, info) in enumerate(available_langs.items()):
                    lang_options.append(info.get('name', code))
                    lang_codes.append(code)
                    if code == current_lang:
                        current_index = i
            except Exception as e:
                st.error(f"❌ Erro na configuração de idiomas: {e}")
                return
            
            # Selectbox para idioma
            if lang_options:  # PROTEÇÃO adicional
                selected_label = st.selectbox(
                    "🌍 Idioma | Language",
                    lang_options,
                    index=current_index,
                    key='language_selector'
                )
                
                # Atualizar idioma se mudou
                try:
                    selected_index = lang_options.index(selected_label)
                    selected_code = lang_codes[selected_index]
                    
                    if selected_code != current_lang:
                        self.set_language(selected_code)
                        st.rerun()
                except (ValueError, IndexError) as e:
                    st.error(f"❌ Erro ao trocar idioma: {e}")
    
    def translate_data_value(self, value):
        """Traduzir valores específicos dos dados"""
        current_lang = self.get_language()
        
        # Mapeamentos de tradução para valores comuns
        value_translations = {
            'pt': {
                '>50K': 'Acima de 50K',
                '<=50K': 'Até 50K',
                'Male': 'Masculino',
                'Female': 'Feminino',
                'Private': 'Privado',
                'Self-emp-not-inc': 'Autônomo (não incorporado)',
                'Self-emp-inc': 'Autônomo (incorporado)',
                'Federal-gov': 'Governo Federal',
                'Local-gov': 'Governo Local',
                'State-gov': 'Governo Estadual',
                'Without-pay': 'Sem Pagamento',
                'Never-worked': 'Nunca Trabalhou',
                'Married-civ-spouse': 'Casado(a)',
                'Never-married': 'Solteiro(a)',
                'Divorced': 'Divorciado(a)',
                'Separated': 'Separado(a)',
                'Widowed': 'Viúvo(a)',
                'HS-grad': 'Ensino Médio',
                'Some-college': 'Faculdade Incompleta',
                'Bachelors': 'Bacharelado',
                'Masters': 'Mestrado',
                'Doctorate': 'Doutorado',
                'Prof-school': 'Escola Profissional'
            },
            'en': {
                # Valores já em inglês - retornar como estão
            }
        }
        
        translations = value_translations.get(current_lang, {})
        return translations.get(value, value)