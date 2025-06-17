"""
Página de Regras de Associação
Análise de Mineração de Dados - APRIORI, FP-GROWTH, ECLAT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime

def show_association_rules_page(data):
    """Página de regras de associação"""
    st.title("🔗 Regras de Associação")
    st.markdown("### Mineração de Padrões - APRIORI • FP-GROWTH • ECLAT")
    
    # Introdução contextual
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
    ">
        <p style="margin: 0; color: #495057;">
            🔗 <strong>Mineração de Regras de Associação</strong> identifica padrões frequentes e relacionamentos 
            entre variáveis, utilizando algoritmos clássicos como APRIORI, FP-GROWTH e ECLAT.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar disponibilidade dos algoritmos
    algorithms = {
        'APRIORI': 'apriori_rules',
        'FP-GROWTH': 'fp_growth_rules',
        'ECLAT': 'eclat_rules'
    }
    
    available_algorithms = {name: key for name, key in algorithms.items() if key in data and len(data[key]) > 0}
    
    if not available_algorithms:
        show_no_rules_data()
        return
    
    # Comparação geral dos algoritmos
    show_algorithms_comparison(data, available_algorithms)
    
    # Seletor de algoritmo para análise detalhada
    st.subheader("🔍 Análise Detalhada por Algoritmo")
    
    selected_algorithm = st.selectbox(
        "Selecionar algoritmo para análise:",
        list(available_algorithms.keys()),
        help="Escolha o algoritmo para ver análise detalhada das regras"
    )
    
    if selected_algorithm:
        algorithm_key = available_algorithms[selected_algorithm]
        show_detailed_algorithm_analysis(data[algorithm_key], selected_algorithm)
    
    # Análise comparativa se múltiplos algoritmos disponíveis
    if len(available_algorithms) > 1:
        show_comparative_analysis(data, available_algorithms)
    
    # Insights e recomendações
    show_association_insights(data, available_algorithms)

def show_no_rules_data():
    """Mostrar aviso quando dados de regras não estão disponíveis"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 2rem 0;
        border: 1px solid #ffecb5;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🔗</div>
        <h3 style="color: #856404; margin-bottom: 1rem;">Regras de Associação Não Executadas</h3>
        <p style="color: #856404; margin-bottom: 1.5rem;">
            Execute o pipeline completo para gerar as regras de associação.
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
    
    # Informações sobre os algoritmos
    st.markdown("#### 📚 Sobre os Algoritmos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **⛏️ APRIORI**
        - Algoritmo clássico
        - Busca em largura
        - Gera candidatos iterativamente
        - Eficiente para datasets pequenos
        """)
    
    with col2:
        st.markdown("""
        **🌳 FP-GROWTH**
        - Estrutura de árvore FP
        - Mais eficiente que APRIORI
        - Sem geração de candidatos
        - Adequado para datasets grandes
        """)
    
    with col3:
        st.markdown("""
        **📊 ECLAT**
        - Busca vertical
        - Intersecção de listas
        - Parallelização eficiente
        - Boa para datasets esparsos
        """)

def show_algorithms_comparison(data, available_algorithms):
    """Comparação geral dos algoritmos"""
    st.subheader("📊 Comparação dos Algoritmos")
    
    # Coletar estatísticas de cada algoritmo
    comparison_data = []
    
    for name, key in available_algorithms.items():
        rules_df = data[key]
        
        stats = {
            'Algoritmo': name,
            'Total de Regras': len(rules_df),
            'Confiança Média': rules_df['confidence'].mean() if 'confidence' in rules_df.columns else 0,
            'Suporte Médio': rules_df['support'].mean() if 'support' in rules_df.columns else 0,
            'Lift Médio': rules_df['lift'].mean() if 'lift' in rules_df.columns else 0
        }
        
        comparison_data.append(stats)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Cards de comparação
    cols = st.columns(len(available_algorithms))
    
    for i, (name, key) in enumerate(available_algorithms.items()):
        with cols[i]:
            rules_count = len(data[key])
            avg_confidence = data[key]['confidence'].mean() if 'confidence' in data[key].columns else 0
            
            # Cor baseada na performance
            if avg_confidence > 0.8:
                color = "#28a745"
            elif avg_confidence > 0.6:
                color = "#ffc107"
            else:
                color = "#17a2b8"
            
            icon = {"APRIORI": "⛏️", "FP-GROWTH": "🌳", "ECLAT": "📊"}.get(name, "🔗")
            
            create_algorithm_card(f"{icon} {name}", rules_count, f"Conf: {avg_confidence:.3f}", color)
    
    # Tabela comparativa detalhada
    st.markdown("#### 📋 Tabela Comparativa")
    
    # Formatar valores para melhor visualização
    comparison_df_formatted = comparison_df.copy()
    for col in ['Confiança Média', 'Suporte Médio', 'Lift Médio']:
        if col in comparison_df_formatted.columns:
            comparison_df_formatted[col] = comparison_df_formatted[col].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(comparison_df_formatted, use_container_width=True)
    
    # Gráfico comparativo
    if len(available_algorithms) > 1:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Total de Regras', 'Confiança Média', 'Suporte Médio'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        algorithms_list = list(available_algorithms.keys())
        
        # Total de regras
        fig.add_trace(
            go.Bar(
                x=algorithms_list,
                y=[len(data[available_algorithms[alg]]) for alg in algorithms_list],
                name="Regras",
                marker_color="#4285f4"
            ),
            row=1, col=1
        )
        
        # Confiança média
        fig.add_trace(
            go.Bar(
                x=algorithms_list,
                y=[data[available_algorithms[alg]]['confidence'].mean() if 'confidence' in data[available_algorithms[alg]].columns else 0 for alg in algorithms_list],
                name="Confiança",
                marker_color="#28a745"
            ),
            row=1, col=2
        )
        
        # Suporte médio
        fig.add_trace(
            go.Bar(
                x=algorithms_list,
                y=[data[available_algorithms[alg]]['support'].mean() if 'support' in data[available_algorithms[alg]].columns else 0 for alg in algorithms_list],
                name="Suporte",
                marker_color="#ffc107"
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            title="Comparação entre Algoritmos",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_algorithm_card(title, rules_count, confidence_text, color):
    """Criar card de algoritmo"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}08);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid {color};
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        min-height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <h4 style="margin: 0; color: #333; font-size: 1rem; font-weight: 600;">
            {title}
        </h4>
        <h2 style="margin: 0.8rem 0; color: {color}; font-size: 2rem; font-weight: bold;">
            {rules_count}
        </h2>
        <p style="margin: 0; color: #666; font-size: 0.85rem; opacity: 0.8;">
            {confidence_text}
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_detailed_algorithm_analysis(rules_df, algorithm_name):
    """Análise detalhada de um algoritmo específico"""
    st.markdown(f"#### 🔍 Análise Detalhada - {algorithm_name}")
    
    if rules_df.empty:
        st.warning(f"⚠️ Nenhuma regra encontrada para {algorithm_name}")
        return
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total de Regras", len(rules_df))
    
    with col2:
        if 'confidence' in rules_df.columns:
            avg_conf = rules_df['confidence'].mean()
            st.metric("🎯 Confiança Média", f"{avg_conf:.3f}")
        else:
            st.metric("🎯 Confiança Média", "N/A")
    
    with col3:
        if 'support' in rules_df.columns:
            avg_supp = rules_df['support'].mean()
            st.metric("📈 Suporte Médio", f"{avg_supp:.3f}")
        else:
            st.metric("📈 Suporte Médio", "N/A")
    
    with col4:
        if 'lift' in rules_df.columns:
            avg_lift = rules_df['lift'].mean()
            st.metric("🚀 Lift Médio", f"{avg_lift:.3f}")
        else:
            st.metric("🚀 Lift Médio", "N/A")
    
    # Top regras
    st.markdown("##### 🏆 Melhores Regras")
    
    # Ordenar por confiança se disponível
    if 'confidence' in rules_df.columns:
        top_rules = rules_df.nlargest(10, 'confidence')
    else:
        top_rules = rules_df.head(10)
    
    # Mostrar top regras em formato mais legível
    if not top_rules.empty:
        display_rules_table(top_rules)
    
    # Visualizações das métricas
    show_metrics_visualizations(rules_df, algorithm_name)

def display_rules_table(rules_df):
    """Exibir tabela de regras formatada"""
    display_df = rules_df.copy()
    
    # Formatar colunas numéricas
    numeric_cols = ['support', 'confidence', 'lift']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
    
    # Renomear colunas para português
    column_mapping = {
        'antecedents': 'Antecedente',
        'consequents': 'Consequente', 
        'support': 'Suporte',
        'confidence': 'Confiança',
        'lift': 'Lift'
    }
    
    display_df = display_df.rename(columns=column_mapping)
    
    st.dataframe(display_df, use_container_width=True)

def show_metrics_visualizations(rules_df, algorithm_name):
    """Mostrar visualizações das métricas"""
    st.markdown("##### 📈 Distribuição das Métricas")
    
    # Verificar quais métricas estão disponíveis
    available_metrics = [col for col in ['support', 'confidence', 'lift'] if col in rules_df.columns]
    
    if not available_metrics:
        st.info("📊 Métricas detalhadas não disponíveis")
        return
    
    # Criar subplots para cada métrica disponível
    cols = st.columns(len(available_metrics))
    
    for i, metric in enumerate(available_metrics):
        with cols[i]:
            # Histograma da métrica
            fig = px.histogram(
                rules_df,
                x=metric,
                nbins=20,
                title=f"Distribuição - {metric.title()}",
                labels={metric: metric.title(), 'count': 'Frequência'}
            )
            
            # Personalizar cores
            colors = {'support': '#4285f4', 'confidence': '#28a745', 'lift': '#ffc107'}
            fig.update_traces(marker_color=colors.get(metric, '#17a2b8'))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot se múltiplas métricas disponíveis
    if len(available_metrics) >= 2:
        st.markdown("##### 🔍 Correlação entre Métricas")
        
        # Seletor de métricas para scatter plot
        col1, col2 = st.columns(2)
        
        with col1:
            x_metric = st.selectbox(
                "Métrica X:",
                available_metrics,
                key=f"x_metric_{algorithm_name}"
            )
        
        with col2:
            y_metrics = [m for m in available_metrics if m != x_metric]
            if y_metrics:
                y_metric = st.selectbox(
                    "Métrica Y:",
                    y_metrics,
                    key=f"y_metric_{algorithm_name}"
                )
                
                # Criar scatter plot
                fig_scatter = px.scatter(
                    rules_df,
                    x=x_metric,
                    y=y_metric,
                    title=f"{y_metric.title()} vs {x_metric.title()}",
                    labels={x_metric: x_metric.title(), y_metric: y_metric.title()},
                    opacity=0.7
                )
                
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)

def show_comparative_analysis(data, available_algorithms):
    """Análise comparativa entre algoritmos"""
    st.subheader("⚖️ Análise Comparativa")
    
    # Criar dataframe combinado para comparação
    comparison_metrics = []
    
    for name, key in available_algorithms.items():
        rules_df = data[key]
        
        if not rules_df.empty:
            metrics = {
                'Algoritmo': name,
                'Regras': len(rules_df),
                'Confiança_Min': rules_df['confidence'].min() if 'confidence' in rules_df.columns else 0,
                'Confiança_Max': rules_df['confidence'].max() if 'confidence' in rules_df.columns else 0,
                'Confiança_Média': rules_df['confidence'].mean() if 'confidence' in rules_df.columns else 0,
                'Suporte_Média': rules_df['support'].mean() if 'support' in rules_df.columns else 0,
                'Lift_Média': rules_df['lift'].mean() if 'lift' in rules_df.columns else 0
            }
            comparison_metrics.append(metrics)
    
    if comparison_metrics:
        comp_df = pd.DataFrame(comparison_metrics)
        
        # Gráfico radar para comparação
        create_radar_chart(comp_df)
        
        # Análise de overlap de regras
        if len(available_algorithms) == 2:
            show_rules_overlap_analysis(data, available_algorithms)

def create_radar_chart(comparison_df):
    """Criar gráfico radar para comparação"""
    st.markdown("#### 🕸️ Comparação Multidimensional")
    
    if len(comparison_df) < 2:
        st.info("📊 Necessário pelo menos 2 algoritmos para comparação radar")
        return
    
    # Normalizar métricas para escala 0-1
    metrics_cols = ['Confiança_Média', 'Suporte_Média', 'Lift_Média']
    
    fig = go.Figure()
    
    colors = ['#4285f4', '#28a745', '#ffc107']
    
    for i, row in comparison_df.iterrows():
        # Normalizar valores
        normalized_values = []
        for col in metrics_cols:
            max_val = comparison_df[col].max()
            min_val = comparison_df[col].min()
            if max_val > min_val:
                normalized = (row[col] - min_val) / (max_val - min_val)
            else:
                normalized = 1.0
            normalized_values.append(normalized)
        
        # Adicionar o primeiro valor no final para fechar o radar
        normalized_values.append(normalized_values[0])
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=metrics_cols + [metrics_cols[0]],
            fill='toself',
            name=row['Algoritmo'],
            line_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Comparação Normalizada das Métricas",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_rules_overlap_analysis(data, available_algorithms):
    """Análise de sobreposição de regras entre algoritmos"""
    st.markdown("#### 🔄 Análise de Sobreposição")
    
    if len(available_algorithms) != 2:
        return
    
    alg_names = list(available_algorithms.keys())
    alg1_data = data[available_algorithms[alg_names[0]]]
    alg2_data = data[available_algorithms[alg_names[1]]]
    
    # Comparar regras (simplificado)
    alg1_rules = len(alg1_data)
    alg2_rules = len(alg2_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(f"📊 {alg_names[0]}", alg1_rules)
    
    with col2:
        st.metric(f"📊 {alg_names[1]}", alg2_rules)
    
    with col3:
        # Estimativa de overlap (simplificada)
        overlap_estimate = min(alg1_rules, alg2_rules) * 0.7  # Estimativa
        st.metric("🔄 Overlap Estimado", f"{overlap_estimate:.0f}")
    
    st.info("💡 Análise de overlap detalhada requer estrutura de dados específica")

def show_association_insights(data, available_algorithms):
    """Mostrar insights e recomendações"""
    st.subheader("💡 Insights e Recomendações")
    
    insights = []
    
    # Análise geral
    total_rules = sum(len(data[key]) for key in available_algorithms.values())
    
    if total_rules == 0:
        insights.append({
            "icon": "🔴",
            "title": "Nenhuma Regra Encontrada",
            "message": "Nenhum algoritmo gerou regras. Verifique os parâmetros de suporte e confiança mínimos.",
            "type": "error"
        })
    elif total_rules < 10:
        insights.append({
            "icon": "🟡",
            "title": "Poucas Regras Identificadas",
            "message": f"Apenas {total_rules} regras encontradas. Considere reduzir os thresholds de suporte e confiança.",
            "type": "warning"
        })
    else:
        insights.append({
            "icon": "🟢",
            "title": "Boa Quantidade de Regras",
            "message": f"{total_rules} regras identificadas. Boa base para análise de padrões.",
            "type": "success"
        })
    
    # Análise de qualidade
    all_confidences = []
    for key in available_algorithms.values():
        if 'confidence' in data[key].columns:
            all_confidences.extend(data[key]['confidence'].tolist())
    
    if all_confidences:
        avg_confidence = np.mean(all_confidences)
        
        if avg_confidence > 0.8:
            insights.append({
                "icon": "🟢",
                "title": "Alta Confiança das Regras",
                "message": f"Confiança média de {avg_confidence:.3f}. Regras altamente confiáveis identificadas.",
                "type": "success"
            })
        elif avg_confidence > 0.6:
            insights.append({
                "icon": "🟡",
                "title": "Confiança Moderada",
                "message": f"Confiança média de {avg_confidence:.3f}. Regras moderadamente confiáveis.",
                "type": "warning"
            })
        else:
            insights.append({
                "icon": "🔴",
                "title": "Baixa Confiança",
                "message": f"Confiança média de {avg_confidence:.3f}. Considere ajustar parâmetros.",
                "type": "error"
            })
    
    # Mostrar insights
    for insight in insights:
        bg_color = {
            "success": "#d4edda",
            "warning": "#fff3cd", 
            "error": "#f8d7da"
        }.get(insight["type"], "#e9ecef")
        
        border_color = {
            "success": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545"
        }.get(insight["type"], "#6c757d")
        
        st.markdown(f"""
        <div style="
            background: {bg_color};
            padding: 1.2rem;
            border-radius: 10px;
            border-left: 4px solid {border_color};
            margin: 1rem 0;
            display: flex;
            align-items: flex-start;
        ">
            <div style="font-size: 1.5rem; margin-right: 1rem; margin-top: 0.2rem;">
                {insight['icon']}
            </div>
            <div>
                <h4 style="margin: 0 0 0.5rem 0; color: #333;">
                    {insight['title']}
                </h4>
                <p style="margin: 0; color: #555; line-height: 1.5;">
                    {insight['message']}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recomendações técnicas
    st.markdown("#### 🔧 Recomendações Técnicas")
    
    recommendations = [
        "📊 **Ajuste de Parâmetros**: Experimente diferentes valores de suporte e confiança mínimos",
        "🔍 **Análise de Qualidade**: Foque nas regras com maior lift para insights mais valiosos",
        "📈 **Interpretação**: Analise o contexto de negócio das regras mais confiáveis",
        "⚖️ **Comparação**: Use múltiplos algoritmos para validar descobertas",
        "🎯 **Filtragem**: Remova regras redundantes ou óbvias para melhor análise"
    ]
    
    for rec in recommendations:
        st.markdown(f"- {rec}")
    
    # Próximos passos
    st.markdown("#### 🚀 Próximos Passos")
    
    next_steps = [
        "1. **Validação**: Teste as regras em dados novos",
        "2. **Implementação**: Aplique insights em decisões de negócio", 
        "3. **Monitoramento**: Acompanhe a performance das regras ao longo do tempo",
        "4. **Refinamento**: Ajuste parâmetros baseado nos resultados obtidos"
    ]
    
    for step in next_steps:
        st.markdown(step)