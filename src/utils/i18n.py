"""
🌍 Sistema de Internacionalização (i18n)
Gerenciamento completo de traduções JSON
"""

import streamlit as st
import json
from pathlib import Path
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class I18nSystem:
    """Sistema de internacionalização com arquivos JSON separados"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_language = 'pt'  # Idioma padrão
        self.translations = {}
        self.fallback_translations = {}
        
        # Carregar traduções dos arquivos JSON
        self._load_translations()

    def _load_translations(self):
        """Carregar traduções dos arquivos JSON da pasta translate/"""
        try:
            # Procurar pasta translate/ (baseada na estrutura do projeto)
            translations_paths = [
                Path("translate"),
                Path("translations"), 
                Path("./translate"),
                Path("../translate")
            ]
            
            translations_dir = None
            for path in translations_paths:
                if path.exists() and any(path.glob("*.json")):
                    translations_dir = path
                    self.logger.info(f"📁 Pasta de traduções encontrada: {translations_dir}")
                    break
            
            if not translations_dir:
                self.logger.warning("📁 Pasta translate/ não encontrada, usando traduções padrão")
                self._load_default_translations()
                return
            
            # Carregar todos os arquivos JSON da pasta
            for json_file in translations_dir.glob("*.json"):
                language = json_file.stem  # Nome do arquivo sem extensão
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        self.translations[language] = json.load(f)
                    self.logger.info(f"✅ Traduções {language.upper()} carregadas de {json_file.name}")
                except Exception as e:
                    self.logger.error(f"❌ Erro ao carregar {json_file.name}: {e}")
            
            # Usar português como fallback se disponível
            if 'pt' in self.translations:
                self.fallback_translations = self._flatten_dict(self.translations['pt'])
                self.logger.info("✅ Fallback em português configurado")
            elif self.translations:
                # Usar primeiro idioma disponível como fallback
                first_lang = list(self.translations.keys())[0]
                self.fallback_translations = self._flatten_dict(self.translations[first_lang])
                self.logger.info(f"✅ Fallback em {first_lang} configurado")
            else:
                self.logger.warning("⚠️ Nenhuma tradução carregada, usando padrão")
                self._load_default_translations()
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar traduções: {e}")
            self._load_default_translations()

    def _load_default_translations(self):
        """Carregar traduções padrão se os arquivos JSON não existirem"""
        self.translations = {
            'pt': {
                # Navegação
                'navigation': {
                    'overview': 'Visão Geral',
                    'exploratory': 'Análise Exploratória',
                    'models': 'Modelos ML',
                    'clustering': 'Clustering',
                    'association_rules': 'Regras de Associação',
                    'prediction': 'Predição',
                    'metrics': 'Métricas',
                    'reports': 'Relatórios',
                    'admin': 'Administração',
                    'data_status': 'Status dos Dados'
                },
                
                # Authentication
                'auth': {
                    'login': 'Login',
                    'username': 'Nome de usuário',
                    'password': 'Senha',
                    'login_button': 'Entrar',
                    'login_required': 'Login necessário',
                    'admin_required': 'Acesso restrito a administradores',
                    'invalid_credentials': 'Credenciais inválidas',
                    'logout': 'Sair',
                    'welcome': 'Bem-vindo'
                },
                
                # Association Rules
                'association': {
                    'title': 'Análise de Regras de Associação',
                    'comparison': 'Comparação de Algoritmos',
                    'apriori': 'Algoritmo Apriori',
                    'fp_growth': 'Algoritmo FP-Growth',
                    'eclat': 'Algoritmo Eclat',
                    'visualizations': 'Visualizações',
                    'rules_found': 'Regras Encontradas',
                    'avg_confidence': 'Confiança Média',
                    'avg_lift': 'Lift Médio',
                    'max_confidence': 'Confiança Máxima',
                    'execution_status': 'Status de Execução',
                    'min_confidence': 'Confiança Mínima',
                    'min_lift': 'Lift Mínimo',
                    'top_rules': 'Top Regras',
                    'no_rules': 'Nenhuma regra encontrada',
                    'total_rules': 'Total de Regras',
                    'best_algorithm': 'Melhor Algoritmo',
                    'general_confidence': 'Confiança Geral',
                    'algorithms_ok': 'Algoritmos OK'
                }
            },
            'en': {
                # Navigation
                'navigation': {
                    'overview': 'Overview',
                    'exploratory': 'Exploratory Analysis',
                    'models': 'ML Models',
                    'clustering': 'Clustering',
                    'association_rules': 'Association Rules',
                    'prediction': 'Prediction',
                    'metrics': 'Metrics',
                    'reports': 'Reports',
                    'admin': 'Administration',
                    'data_status': 'Data Status'
                },
                
                # Authentication
                'auth': {
                    'login': 'Login',
                    'username': 'Username',
                    'password': 'Password',
                    'login_button': 'Sign In',
                    'login_required': 'Login required',
                    'admin_required': 'Admin access required',
                    'invalid_credentials': 'Invalid credentials',
                    'logout': 'Logout',
                    'welcome': 'Welcome'
                },
                
                # Association Rules
                'association': {
                    'title': 'Association Rules Analysis',
                    'comparison': 'Algorithm Comparison',
                    'apriori': 'Apriori Algorithm',
                    'fp_growth': 'FP-Growth Algorithm',
                    'eclat': 'Eclat Algorithm',
                    'visualizations': 'Visualizations',
                    'rules_found': 'Rules Found',
                    'avg_confidence': 'Average Confidence',
                    'avg_lift': 'Average Lift',
                    'max_confidence': 'Max Confidence',
                    'execution_status': 'Execution Status',
                    'min_confidence': 'Min Confidence',
                    'min_lift': 'Min Lift',
                    'top_rules': 'Top Rules',
                    'no_rules': 'No rules found',
                    'total_rules': 'Total Rules',
                    'best_algorithm': 'Best Algorithm',
                    'general_confidence': 'General Confidence',
                    'algorithms_ok': 'Algorithms OK'
                }
            }
        }
        
        self.fallback_translations = self._flatten_dict(self.translations.get('pt', {}))

    def set_language(self, language: str):
        """Definir idioma atual"""
        if language in self.translations:
            self.current_language = language
            self.logger.info(f"🌍 Idioma alterado para: {language}")
        else:
            self.logger.warning(f"⚠️ Idioma '{language}' não disponível")
    
    def get_language(self) -> str:
        """Obter idioma atual"""
        return self.current_language
    
    def get_available_languages(self) -> list:
        """Obter lista de idiomas disponíveis"""
        return list(self.translations.keys())

    def t(self, key: str, default: Optional[str] = None) -> str:
        """Traduzir chave (suporta estrutura aninhada dos JSONs)"""
        try:
            # Tentar idioma atual
            current_translations = self.translations.get(self.current_language, {})
            
            # Buscar por chave aninhada (ex: 'navigation.overview')
            if '.' in key:
                keys = key.split('.')
                value = current_translations
                
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        # Tentar fallback
                        flattened_fallback = self.fallback_translations
                        fallback_value = flattened_fallback.get(key)
                        return fallback_value or default or key
                
                return str(value) if not isinstance(value, dict) else (default or key)
            
            # Chave simples - tentar primeiro nível
            translation = current_translations.get(key)
            if translation:
                return translation
            
            # Fallback
            fallback = self.fallback_translations.get(key)
            if fallback:
                return fallback
            
            # Retornar default ou a própria chave
            return default or key
            
        except Exception as e:
            self.logger.error(f"❌ Erro na tradução da chave '{key}': {e}")
            return default or key

    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '.') -> dict:
        """Achatar dicionário aninhado para busca mais eficiente"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def debug_translations(self):
        """Debug das traduções carregadas"""
        self.logger.info("🔍 DEBUG DE TRADUÇÕES")
        self.logger.info(f"   Idioma atual: {self.current_language}")
        self.logger.info(f"   Idiomas disponíveis: {list(self.translations.keys())}")
        
        for lang, translations in self.translations.items():
            flat_translations = self._flatten_dict(translations)
            self.logger.info(f"   {lang.upper()}: {len(flat_translations)} chaves")