"""
Página de Visão Geral - Overview
Dashboard de Análise Salarial Científica
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def show_overview_page(data):
    """Página de visão geral"""
    st.title("📊 Visão Geral")
    st.markdown("### Dashboard de Análise Salarial Científica")
    
    # Subtitle com informações contextuais
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
    ">
        <p style="margin: 0; color: #495057;">
            📈 <strong>Sistema integrado de análise científica</strong> utilizando algoritmos de clustering (DBSCAN) 
            e mineração de regras de associação (APRIORI, FP-GROWTH, ECLAT) para análise salarial.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not data:
        show_no_data_warning()
        return
    
    # Status dos algoritmos
    show_algorithm_status(data)
    
    # Métricas principais
    show_main_metrics(data)
    
    # Análise dos dados originais
    if 'original' in data:
        show_dataset_analysis(data['original'])
    
    # Resumo executivo
    show_executive_summary(data)

def show_no_data_warning():
    """Mostrar aviso quando dados não estão disponíveis"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 2rem 0;
        border: 1px solid #ffecb5;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
        <h3 style="color: #856404; margin-bottom: 1rem;">Dados Não Encontrados</h3>
        <p style="color: #856404; margin-bottom: 1.5rem;">
            Para visualizar a análise completa, execute o pipeline de dados.
        </p>
        <div style="
            background: rgba(133, 100, 4, 0.1);
            padding: 1rem;
            border-radius: 8px;
            font-family: monospace;
            color: #856404;
            font-weight: bold;
        ">
            python main.py
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    💡 **O pipeline irá executar:**
    - Carregamento e preprocessamento dos dados
    - Análise de clustering com DBSCAN
    - Mineração de regras de associação (APRIORI, FP-GROWTH, ECLAT)
    - Geração de métricas e visualizações
    """)

def show_algorithm_status(data):
    """Mostrar status dos algoritmos com cards modernos"""
    st.subheader("🔄 Status dos Algoritmos")
    
    algorithms = {
        "DBSCAN": {
            "key": "dbscan_results",
            "icon": "🎯",
            "description": "Clustering baseado em densidade"
        },
        "APRIORI": {
            "key": "apriori_rules",
            "icon": "⛏️",
            "description": "Mineração clássica de regras"
        },
        "FP-GROWTH": {
            "key": "fp_growth_rules",
            "icon": "🌳",
            "description": "Algoritmo de árvore FP"
        },
        "ECLAT": {
            "key": "eclat_rules",
            "icon": "📊",
            "description": "Busca vertical de itemsets"
        }
    }
    
    cols = st.columns(4)
    
    for i, (name, info) in enumerate(algorithms.items()):
        with cols[i]:
            is_active = info["key"] in data and len(data[info["key"]]) > 0
            count = len(data[info["key"]]) if is_active else 0
            
            # Cores baseadas no status
            bg_color = "#d4edda" if is_active else "#f8d7da"
            border_color = "#28a745" if is_active else "#dc3545"
            text_color = "#155724" if is_active else "#721c24"
            status_icon = "✅" if is_active else "❌"
            
            st.markdown(f"""
            <div style="
                background: {bg_color};
                padding: 1.2rem;
                border-radius: 10px;
                border: 2px solid {border_color};
                text-align: center;
                margin-bottom: 1rem;
                min-height: 140px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            ">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">
                    {info['icon']}
                </div>
                <h4 style="margin: 0; color: {text_color}; font-weight: bold;">
                    {status_icon} {name}
                </h4>
                <p style="margin: 0.5rem 0; color: {text_color}; font-size: 0.85rem;">
                    {info['description']}
                </p>
                <div style="color: {text_color}; font-weight: bold; font-size: 1.1rem;">
                    {'Resultados: ' + str(count) if is_active else 'Não executado'}
                </div>
            </div>
            """, unsafe_allow_html=True)

def show_main_metrics(data):
    """Mostrar métricas principais em cards elegantes"""
    st.subheader("📈 Métricas Principais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Número de registros
        if 'original' in data:
            records = len(data['original'])
            create_metric_card("📋 Registros", f"{records:,}", "Total de observações", "#4285f4")
        else:
            create_metric_card("📋 Registros", "N/A", "Dados não disponíveis", "#6c757d")
    
    with col2:
        # Número de colunas
        if 'original' in data:
            columns = len(data['original'].columns)
            create_metric_card("📊 Colunas", columns, "Variáveis no dataset", "#28a745")
        else:
            create_metric_card("📊 Colunas", "N/A", "Dados não disponíveis", "#6c757d")
    
    with col3:
        # Idade média
        if 'original' in data and 'age' in data['original'].columns:
            avg_age = data['original']['age'].mean()
            create_metric_card("👤 Idade Média", f"{avg_age:.1f}", "Anos", "#ffc107")
        else:
            create_metric_card("👤 Idade Média", "N/A", "Dados não disponíveis", "#6c757d")
    
    with col4:
        # Grau acadêmico com melhor remuneração
        if 'original' in data and 'education' in data['original'].columns and 'salary' in data['original'].columns:
            best_education = get_best_paid_education(data['original'])
            create_metric_card("🎓 Melhor Grau", best_education['education'], f"{best_education['percentage']:.1f}% >50K", "#17a2b8")
        else:
            create_metric_card("🎓 Melhor Grau", "N/A", "Dados não disponíveis", "#6c757d")

def get_best_paid_education(df):
    """Encontrar o grau acadêmico com melhor remuneração"""
    try:
        # Calcular percentual de salários >50K por nível educacional
        education_salary = df.groupby('education')['salary'].apply(
            lambda x: (x == '>50K').mean() * 100
        ).sort_values(ascending=False)
        
        if len(education_salary) > 0:
            best_education = education_salary.index[0]
            best_percentage = education_salary.iloc[0]
            
            # Mapear nomes para versões mais legíveis
            education_mapping = {
                'Doctorate': 'Doutorado',
                'Masters': 'Mestrado', 
                'Prof-school': 'Escola Prof.',
                'Bachelors': 'Bacharelado',
                'Assoc-acdm': 'Assoc. Acad.',
                'Assoc-voc': 'Assoc. Tec.',
                'Some-college': 'Sup. Incomp.',
                'HS-grad': 'Ens. Médio',
                '11th': '11º ano',
                '10th': '10º ano',
                '9th': '9º ano',
                '7th-8th': '7º-8º ano',
                '5th-6th': '5º-6º ano',
                '1st-4th': '1º-4º ano',
                'Preschool': 'Pré-escola'
            }
            
            display_name = education_mapping.get(best_education, best_education)
            
            return {
                'education': display_name,
                'percentage': best_percentage,
                'original_name': best_education
            }
        else:
            return {
                'education': 'N/A',
                'percentage': 0,
                'original_name': 'N/A'
            }
    except Exception as e:
        return {
            'education': 'Erro',
            'percentage': 0,
            'original_name': 'Error'
        }

def create_metric_card(title, value, description, color):
    """Criar card de métrica estilizado"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}08);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid {color};
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <h4 style="margin: 0; color: #333; font-size: 0.9rem; font-weight: 600;">
            {title}
        </h4>
        <h2 style="margin: 0.8rem 0; color: {color}; font-size: 2rem; font-weight: bold;">
            {value}
        </h2>
        <p style="margin: 0; color: #666; font-size: 0.8rem; opacity: 0.8;">
            {description}
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_dataset_analysis(df):
    """Análise detalhada do dataset"""
    st.subheader("📊 Análise do Dataset")
    
    # Cards de informações básicas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📋 Informações Básicas
        """)
        
        info_data = {
            "📏 Dimensões": f"{len(df):,} × {len(df.columns)}",
            "💾 Memória": f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
            "🔍 Valores Nulos": f"{df.isnull().sum().sum():,}",
            "📊 Completude": f"{((df.count().sum() / (len(df) * len(df.columns))) * 100):.1f}%"
        }
        
        for label, value in info_data.items():
            st.markdown(f"""
            <div style="
                background: rgba(102, 126, 234, 0.1);
                padding: 0.8rem;
                border-radius: 8px;
                margin: 0.3rem 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <span style="font-weight: 500;">{label}</span>
                <span style="color: #667eea; font-weight: bold;">{value}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Análise educacional detalhada
        if 'education' in df.columns and 'salary' in df.columns:
            st.markdown("#### 🎓 Análise Educacional")
            
            # Top 3 graus com melhor remuneração
            education_salary = df.groupby('education')['salary'].apply(
                lambda x: (x == '>50K').mean() * 100
            ).sort_values(ascending=False)
            
            top_3_education = education_salary.head(3)
            
            education_mapping = {
                'Doctorate': 'Doutorado',
                'Masters': 'Mestrado', 
                'Prof-school': 'Escola Prof.',
                'Bachelors': 'Bacharelado',
                'Assoc-acdm': 'Assoc. Acad.',
                'Assoc-voc': 'Assoc. Tec.',
                'Some-college': 'Sup. Incomp.',
                'HS-grad': 'Ens. Médio'
            }
            
            for i, (edu, pct) in enumerate(top_3_education.items(), 1):
                display_name = education_mapping.get(edu, edu)
                color = ["#28a745", "#ffc107", "#17a2b8"][i-1]
                
                st.markdown(f"""
                <div style="
                    background: rgba(40, 167, 69, 0.1);
                    padding: 0.8rem;
                    border-radius: 8px;
                    margin: 0.3rem 0;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    border-left: 3px solid {color};
                ">
                    <span style="font-weight: 500;">{i}º {display_name}</span>
                    <span style="color: {color}; font-weight: bold;">{pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Informações educacionais não disponíveis no dataset")
    
    # Análise de idade se disponível
    if 'age' in df.columns:
        st.markdown("#### 👤 Distribuição Etária")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Estatísticas de idade
            age_stats = {
                "Idade Mínima": df['age'].min(),
                "Idade Máxima": df['age'].max(),
                "Idade Média": df['age'].mean(),
                "Mediana": df['age'].median(),
                "Desvio Padrão": df['age'].std()
            }
            
            for stat, value in age_stats.items():
                st.markdown(f"- **{stat}**: {value:.1f} anos")
        
        with col2:
            # Faixas etárias
            df_temp = df.copy()
            df_temp['faixa_etaria'] = pd.cut(
                df_temp['age'], 
                bins=[0, 25, 35, 45, 55, 100], 
                labels=['18-25', '26-35', '36-45', '46-55', '55+']
            )
            
            faixa_counts = df_temp['faixa_etaria'].value_counts().sort_index()
            
            for faixa, count in faixa_counts.items():
                pct = (count / len(df)) * 100
                st.markdown(f"- **{faixa} anos**: {count:,} ({pct:.1f}%)")
    
    # Gráfico de distribuição salarial por educação
    if 'salary' in df.columns and 'education' in df.columns:
        st.markdown("#### 📈 Distribuição Salarial por Nível Educacional")
        
        # Criar cross-tabulation
        crosstab = pd.crosstab(df['education'], df['salary'], normalize='index') * 100
        crosstab = crosstab.sort_values('>50K', ascending=False)
        
        # Gráfico de barras horizontal
        fig_education = px.bar(
            crosstab.reset_index(),
            x='>50K',
            y='education',
            orientation='h',
            title="Percentual de Salários >50K por Nível Educacional",
            labels={'education': 'Nível Educacional', '>50K': 'Percentual com Salário >50K (%)'},
            color='>50K',
            color_continuous_scale='Blues'
        )
        
        fig_education.update_layout(
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        
        st.plotly_chart(fig_education, use_container_width=True)
    
    # Análise de correlação idade vs salário se disponível
    if 'age' in df.columns and 'salary' in df.columns:
        st.markdown("#### 📊 Idade vs Faixa Salarial")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot idade por salário
            fig_box = px.box(
                df,
                x='salary',
                y='age',
                title="Distribuição de Idade por Faixa Salarial"
            )
            fig_box.update_layout(height=350)
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Idade média por faixa salarial
            age_by_salary = df.groupby('salary')['age'].agg(['mean', 'median', 'std']).round(1)
            
            st.markdown("**Estatísticas por Faixa Salarial:**")
            
            for salary_range in age_by_salary.index:
                stats = age_by_salary.loc[salary_range]
                st.markdown(f"""
                **{salary_range}:**
                - Média: {stats['mean']:.1f} anos
                - Mediana: {stats['median']:.1f} anos
                - Desvio: {stats['std']:.1f} anos
                """)

def show_executive_summary(data):
    """Resumo executivo do sistema"""
    st.subheader("📋 Resumo Executivo")
    
    # Calcular estatísticas do sistema
    total_algorithms = 4
    active_algorithms = sum(1 for key in ['dbscan_results', 'apriori_rules', 'fp_growth_rules', 'eclat_rules'] if key in data and len(data[key]) > 0)
    system_health = (active_algorithms / total_algorithms) * 100
    
    # Determinar status geral
    if system_health >= 75:
        status_color = "#28a745"
        status_icon = "🟢"
        status_text = "Excelente"
    elif system_health >= 50:
        status_color = "#ffc107"
        status_icon = "🟡"
        status_text = "Bom"
    else:
        status_color = "#dc3545"
        status_icon = "🔴"
        status_text = "Requer Atenção"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {status_color}15, {status_color}08);
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid {status_color};
        margin: 1rem 0;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div style="font-size: 2rem; margin-right: 1rem;">{status_icon}</div>
            <div>
                <h3 style="margin: 0; color: {status_color};">
                    Status do Sistema: {status_text}
                </h3>
                <p style="margin: 0.5rem 0; color: #666;">
                    {active_algorithms}/{total_algorithms} algoritmos ativos • {system_health:.0f}% de completude
                </p>
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.7); padding: 1rem; border-radius: 8px;">
            <p style="margin: 0; color: #333; line-height: 1.6;">
                <strong>🎯 Status Atual:</strong> O sistema de análise científica está 
                {"funcionando adequadamente" if system_health >= 50 else "com problemas"}
                com {active_algorithms} de {total_algorithms} algoritmos em execução.
                {"Todos os componentes principais estão operacionais." if system_health >= 75 else 
                 "A maioria dos componentes está funcional." if system_health >= 50 else
                 "Vários componentes necessitam de atenção."}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Próximos passos
    if system_health < 100:
        st.markdown("#### 🚀 Próximos Passos Recomendados")
        
        recommendations = []
        
        if 'dbscan_results' not in data:
            recommendations.append("🎯 Executar análise de clustering DBSCAN")
        
        if 'apriori_rules' not in data:
            recommendations.append("⛏️ Gerar regras de associação APRIORI")
            
        if 'fp_growth_rules' not in data:
            recommendations.append("🌳 Implementar algoritmo FP-GROWTH")
            
        if 'eclat_rules' not in data:
            recommendations.append("📊 Executar mineração ECLAT")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        if recommendations:
            st.info("💡 Execute `python main.py` para completar todas as análises pendentes.")