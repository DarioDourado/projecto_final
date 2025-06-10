"""
📋 Página de Regras de Associação
Análise de padrões e regras de associação nos dados
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

def show_association_rules_page(data, i18n):
    """Página principal de regras de associação"""
    from src.components.navigation import show_page_header
    
    show_page_header(
        i18n.t('navigation.association_rules', 'Regras de Associação'),
        i18n.t('association.subtitle', 'Descoberta de padrões e relações nos dados'),
        "📋"
    )
    
    df = data.get('df')
    if df is None or len(df) == 0:
        st.warning(i18n.t('messages.pipeline_needed', 'Execute pipeline primeiro'))
        return
    
    # Procurar arquivo de regras
    rules_files = _find_rules_files()
    
    if rules_files:
        _show_rules_analysis(rules_files, i18n)
    else:
        _show_rules_placeholder(i18n)

def _find_rules_files():
    """Encontrar arquivos de regras de associação"""
    search_paths = [
        Path("output/analysis"),
        Path("analysis"),
        Path(".")
    ]
    
    rules_files = []
    
    for path in search_paths:
        if path.exists():
            # Procurar arquivos com padrões relacionados a regras
            patterns = [
                "*rule*.csv",
                "*association*.csv", 
                "*frequent*.csv",
                "*pattern*.csv"
            ]
            
            for pattern in patterns:
                rules_files.extend(list(path.glob(pattern)))
    
    return rules_files

def _show_rules_analysis(rules_files, i18n):
    """Mostrar análise das regras encontradas"""
    st.subheader(f"📊 {i18n.t('association.found_rules', 'Regras Encontradas')}")
    
    for file in rules_files:
        with st.expander(f"📄 {file.name}"):
            try:
                rules_df = pd.read_csv(file)
                
                # Mostrar informações básicas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(f"{i18n.t('association.total_rules', 'Total de Regras')}", len(rules_df))
                
                with col2:
                    if 'confidence' in rules_df.columns:
                        avg_confidence = rules_df['confidence'].mean()
                        st.metric(f"{i18n.t('association.avg_confidence', 'Confiança Média')}", f"{avg_confidence:.3f}")
                
                with col3:
                    if 'support' in rules_df.columns:
                        avg_support = rules_df['support'].mean()
                        st.metric(f"{i18n.t('association.avg_support', 'Suporte Médio')}", f"{avg_support:.3f}")
                
                # Mostrar tabela
                st.dataframe(rules_df, use_container_width=True)
                
                # Gráficos se as colunas existirem
                if 'support' in rules_df.columns and 'confidence' in rules_df.columns:
                    _create_rules_scatter_plot(rules_df, i18n)
                
                if 'lift' in rules_df.columns:
                    _create_lift_chart(rules_df, i18n)
                
            except Exception as e:
                st.error(f"❌ {i18n.t('messages.error', 'Erro')} ao carregar {file.name}: {e}")

def _create_rules_scatter_plot(rules_df, i18n):
    """Criar gráfico de dispersão das regras"""
    fig = px.scatter(
        rules_df, 
        x='support', 
        y='confidence',
        hover_data=['antecedents', 'consequents'] if 'antecedents' in rules_df.columns else None,
        title=f"📊 {i18n.t('association.support_vs_confidence', 'Suporte vs Confiança das Regras')}",
        template="plotly_white"
    )
    
    fig.update_layout(
        xaxis_title=i18n.t('association.support', 'Suporte'),
        yaxis_title=i18n.t('association.confidence', 'Confiança')
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _create_lift_chart(rules_df, i18n):
    """Criar gráfico de lift"""
    # Top 20 regras por lift
    top_rules = rules_df.nlargest(20, 'lift')
    
    fig = px.bar(
        top_rules,
        x='lift',
        y=range(len(top_rules)),
        orientation='h',
        title=f"🚀 {i18n.t('association.top_lift', 'Top 20 Regras por Lift')}",
        template="plotly_white"
    )
    
    fig.update_layout(
        yaxis_title=i18n.t('association.rules', 'Regras'),
        xaxis_title=i18n.t('association.lift', 'Lift')
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _show_rules_placeholder(i18n):
    """Mostrar placeholder quando não há regras"""
    st.info(i18n.t('association.no_rules', 'Execute o pipeline para gerar regras de associação'))
    
    # Informações educativas sobre regras de associação
    with st.expander(f"❓ {i18n.t('association.what_are_rules', 'O que são Regras de Associação?')}"):
        st.markdown(f"""
        ### 📚 {i18n.t('association.explanation_title', 'Regras de Associação')}
        
        {i18n.t('association.explanation', '''
        As regras de associação são uma técnica de mineração de dados que identifica 
        relacionamentos frequentes entre diferentes variáveis nos dados.
        ''')}
        
        #### 📊 {i18n.t('association.metrics_title', 'Métricas Principais')}:
        
        - **{i18n.t('association.support', 'Suporte')}**: {i18n.t('association.support_desc', 'Frequência de ocorrência da regra')}
        - **{i18n.t('association.confidence', 'Confiança')}**: {i18n.t('association.confidence_desc', 'Probabilidade condicional da regra')}
        - **{i18n.t('association.lift', 'Lift')}**: {i18n.t('association.lift_desc', 'Medida de interesse da regra')}
        
        #### 🎯 {i18n.t('association.applications_title', 'Aplicações')}:
        
        - {i18n.t('association.app1', 'Análise de comportamento salarial')}
        - {i18n.t('association.app2', 'Identificação de padrões educacionais')}
        - {i18n.t('association.app3', 'Segmentação de perfis profissionais')}
        """)