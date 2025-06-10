"""
🔍 Sistema de Filtros Multilingual Avançado
Filtros dinâmicos com cache e persistência
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

class FilterComponents:
    """Sistema avançado de filtros multilinguals"""
    
    def __init__(self, i18n_system):
        self.i18n = i18n_system
        self.active_filters = {}
        self.filter_cache = {}
    
    def show_filters_sidebar(self, 
                           df: pd.DataFrame,
                           user_role: str = 'guest',
                           container = None) -> Dict[str, Any]:
        """Mostrar filtros na sidebar baseado no papel do usuário"""
        
        if container is None:
            container = st.sidebar
        
        with container:
            # Header dos filtros
            st.markdown(f"### {self.i18n.t('filters.title', '🔍 Filtros')}")
            
            # Verificar se há filtros ativos
            if 'filters' in st.session_state and st.session_state.filters:
                self._show_active_filters()
            
            # Mostrar filtros baseado no papel
            filters = {}
            
            if user_role in ['admin', 'user']:
                filters.update(self._render_advanced_filters(df))
            else:
                filters.update(self._render_basic_filters(df))
            
            # Botões de ação
            self._render_filter_actions(filters)
            
            return filters
    
    def _show_active_filters(self):
        """Mostrar filtros ativos"""
        with st.expander(f"📋 {self.i18n.t('filters.active', 'Filtros Ativos')}", expanded=False):
            for filter_name, filter_value in st.session_state.filters.items():
                if filter_value is not None and filter_value != []:
                    st.write(f"**{filter_name.title()}:** {filter_value}")
    
    def _render_basic_filters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Renderizar filtros básicos para usuários guest"""
        filters = {}
        
        st.markdown("#### 🎯 Filtros Básicos")
        
        # Filtro de salário (se existir)
        if 'salary' in df.columns:
            salary_options = df['salary'].unique().tolist()
            salary_filter = st.multiselect(
                f"💰 {self.i18n.t('data.salary', 'Salário')}:",
                options=salary_options,
                default=None,
                key='salary_filter'
            )
            if salary_filter:
                filters['salary'] = salary_filter
        
        # Filtro de sexo (se existir)
        if 'sex' in df.columns:
            sex_options = df['sex'].unique().tolist()
            sex_filter = st.multiselect(
                f"👥 {self.i18n.t('data.sex', 'Sexo')}:",
                options=[self.i18n.translate_data_value(opt) for opt in sex_options],
                default=None,
                key='sex_filter'
            )
            if sex_filter:
                # Traduzir de volta para valores originais
                original_values = [opt for opt in sex_options 
                                 if self.i18n.translate_data_value(opt) in sex_filter]
                filters['sex'] = original_values
        
        return filters
    
    def _render_advanced_filters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Renderizar filtros avançados para usuários autenticados"""
        filters = {}
        
        # Filtros básicos primeiro
        filters.update(self._render_basic_filters(df))
        
        st.markdown("#### 🔬 Filtros Avançados")
        
        # Filtro de idade (se existir)
        if 'age' in df.columns:
            age_min = int(df['age'].min())
            age_max = int(df['age'].max())
            
            age_range = st.slider(
                f"🎂 {self.i18n.t('data.age', 'Idade')}:",
                min_value=age_min,
                max_value=age_max,
                value=(age_min, age_max),
                key='age_filter'
            )
            
            if age_range != (age_min, age_max):
                filters['age'] = age_range
        
        # Filtro de educação (se existir)
        if 'education' in df.columns:
            education_options = df['education'].unique().tolist()
            education_filter = st.multiselect(
                f"🎓 {self.i18n.t('data.education', 'Educação')}:",
                options=[self.i18n.translate_data_value(opt) for opt in education_options],
                default=None,
                key='education_filter'
            )
            if education_filter:
                original_values = [opt for opt in education_options 
                                 if self.i18n.translate_data_value(opt) in education_filter]
                filters['education'] = original_values
        
        # Filtro de classe trabalhadora (se existir)
        if 'workclass' in df.columns:
            workclass_options = df['workclass'].unique().tolist()
            workclass_filter = st.multiselect(
                f"💼 {self.i18n.t('data.workclass', 'Classe Trabalhadora')}:",
                options=[self.i18n.translate_data_value(opt) for opt in workclass_options],
                default=None,
                key='workclass_filter'
            )
            if workclass_filter:
                original_values = [opt for opt in workclass_options 
                                 if self.i18n.translate_data_value(opt) in workclass_filter]
                filters['workclass'] = original_values
        
        # Filtros numéricos avançados
        self._render_numeric_filters(df, filters)
        
        return filters
    
    def _render_numeric_filters(self, df: pd.DataFrame, filters: Dict[str, Any]):
        """Renderizar filtros para colunas numéricas"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remover colunas já filtradas
        excluded_cols = ['age']  # Age já tem filtro específico
        numeric_cols = [col for col in numeric_cols if col not in excluded_cols]
        
        if numeric_cols:
            st.markdown("##### 📊 Filtros Numéricos")
            
            for col in numeric_cols[:3]:  # Limitar a 3 colunas
                col_min = float(df[col].min())
                col_max = float(df[col].max())
                
                if col_min != col_max:  # Evitar slider com valor único
                    col_range = st.slider(
                        f"📈 {col.title()}:",
                        min_value=col_min,
                        max_value=col_max,
                        value=(col_min, col_max),
                        key=f'{col}_filter'
                    )
                    
                    if col_range != (col_min, col_max):
                        filters[col] = col_range
    
    def _render_filter_actions(self, filters: Dict[str, Any]):
        """Renderizar botões de ação dos filtros"""
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(
                f"✅ {self.i18n.t('filters.apply', 'Aplicar')}",
                key='apply_filters',
                use_container_width=True,
                type='primary'
            ):
                st.session_state.filters = filters
                st.success(f"✅ {len(filters)} {self.i18n.t('filters.applied', 'filtros aplicados')}")
                st.rerun()
        
        with col2:
            if st.button(
                f"🗑️ {self.i18n.t('filters.clear', 'Limpar')}",
                key='clear_filters',
                use_container_width=True,
                type='secondary'
            ):
                # Limpar filtros da sessão
                if 'filters' in st.session_state:
                    del st.session_state.filters
                
                # Limpar estado dos widgets
                filter_keys = [key for key in st.session_state.keys() if key.endswith('_filter')]
                for key in filter_keys:
                    del st.session_state[key]
                
                st.success(f"🗑️ {self.i18n.t('filters.cleared', 'Filtros limpos')}")
                st.rerun()
    
    def apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Aplicar filtros ao DataFrame"""
        if not filters:
            return df
        
        filtered_df = df.copy()
        
        try:
            for filter_name, filter_value in filters.items():
                if filter_value is None or filter_value == []:
                    continue
                
                if filter_name in filtered_df.columns:
                    # Filtros categóricos (lista de valores)
                    if isinstance(filter_value, list):
                        filtered_df = filtered_df[filtered_df[filter_name].isin(filter_value)]
                    
                    # Filtros de range (tupla com min, max)
                    elif isinstance(filter_value, tuple) and len(filter_value) == 2:
                        min_val, max_val = filter_value
                        filtered_df = filtered_df[
                            (filtered_df[filter_name] >= min_val) & 
                            (filtered_df[filter_name] <= max_val)
                        ]
                    
                    # Filtros de valor único
                    else:
                        filtered_df = filtered_df[filtered_df[filter_name] == filter_value]
            
            logger.info(f"✅ Filtros aplicados: {len(df)} -> {len(filtered_df)} registos")
            return filtered_df
            
        except Exception as e:
            logger.error(f"❌ Erro ao aplicar filtros: {e}")
            st.error(f"❌ Erro ao aplicar filtros: {e}")
            return df
    
    def get_filter_summary(self, original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> Dict[str, Any]:
        """Obter resumo dos filtros aplicados"""
        return {
            'original_count': len(original_df),
            'filtered_count': len(filtered_df),
            'reduction_pct': ((len(original_df) - len(filtered_df)) / len(original_df)) * 100,
            'filters_applied': len(st.session_state.get('filters', {}))
        }
    
    def show_filter_summary(self, original_df: pd.DataFrame, filtered_df: pd.DataFrame):
        """Mostrar resumo dos filtros na interface"""
        summary = self.get_filter_summary(original_df, filtered_df)
        
        if summary['filters_applied'] > 0:
            st.info(f"""
            🔍 **Filtros Aplicados:** {summary['filters_applied']}  
            📊 **Registos:** {summary['original_count']:,} → {summary['filtered_count']:,}  
            📉 **Redução:** {summary['reduction_pct']:.1f}%
            """)

    def render_smart_filters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Renderizar filtros inteligentes baseados nos dados"""
        filters = {}
        
        st.markdown("#### 🧠 Filtros Inteligentes")
        
        # Detectar colunas categóricas com poucos valores únicos
        categorical_candidates = []
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() <= 10:
                categorical_candidates.append(col)
        
        # Renderizar filtros para as colunas mais relevantes
        for col in categorical_candidates[:5]:  # Máximo 5 filtros
            unique_values = df[col].unique().tolist()
            
            if len(unique_values) <= 10:  # Apenas se tiver poucos valores
                selected_values = st.multiselect(
                    f"🎯 {col.title()}:",
                    options=unique_values,
                    default=None,
                    key=f'smart_{col}_filter'
                )
                
                if selected_values:
                    filters[col] = selected_values
        
        return filters