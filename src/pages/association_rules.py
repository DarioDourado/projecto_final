"""
📋 Página de Regras de Associação
Análise de padrões e regras de associação nos dados
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging

def show_association_rules_page(df, files_status, i18n):
    """Página principal de regras de associação"""
    
    st.header("📋 " + i18n.t('association.title', 'Análise de Regras de Associação'))
    st.markdown("---")
    
    # Verificar se há resultados
    association_files = [f for f in files_status['analysis'] 
                        if any(keyword in f.name.lower() 
                              for keyword in ['apriori', 'fp_growth', 'eclat', 'association'])]
    
    if not association_files:
        st.warning("⚠️ Nenhum resultado de regras de associação encontrado.")
        st.info("💡 Execute o pipeline principal: `python main.py`")
        return
    
    # Tabs principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 " + i18n.t('association.comparison', 'Comparação'), 
        "🔍 " + i18n.t('association.apriori', 'Apriori'), 
        "🌳 " + i18n.t('association.fp_growth', 'FP-Growth'), 
        "🔗 " + i18n.t('association.eclat', 'Eclat'), 
        "📈 " + i18n.t('association.visualizations', 'Visualizações')
    ])
    
    with tab1:
        show_algorithms_comparison(association_files, i18n)
    
    with tab2:
        show_apriori_results(association_files, i18n)
    
    with tab3:
        show_fp_growth_results(association_files, i18n)
    
    with tab4:
        show_eclat_results(association_files, i18n)
    
    with tab5:
        show_association_visualizations(association_files, i18n)

def show_algorithms_comparison(association_files, i18n):
    """Mostrar comparação entre algoritmos"""
    st.subheader("🏆 " + i18n.t('association.comparison', 'Comparação entre Algoritmos'))
    
    # Procurar arquivo de comparação
    comparison_file = None
    for file in association_files:
        if 'comparison' in file.name.lower() or 'algorithms' in file.name.lower():
            comparison_file = file
            break
    
    if comparison_file:
        try:
            comparison_df = pd.read_csv(comparison_file)
            
            # Tabela de comparação principal
            st.markdown("### 📊 Tabela Comparativa")
            
            # Reformatar para melhor visualização
            display_df = comparison_df.copy()
            
            # Renomear colunas para o idioma atual
            column_mapping = {
                'Algorithm': i18n.t('common.algorithm', 'Algoritmo'),
                'Rules_Found': i18n.t('association.rules_found', 'Regras Encontradas'),
                'Avg_Confidence': i18n.t('association.avg_confidence', 'Confiança Média'),
                'Avg_Lift': i18n.t('association.avg_lift', 'Lift Médio'),
                'Max_Confidence': i18n.t('association.max_confidence', 'Confiança Máxima'),
                'Execution_Status': i18n.t('association.execution_status', 'Status')
            }
            
            # Renomear apenas colunas que existem
            existing_columns = {k: v for k, v in column_mapping.items() if k in display_df.columns}
            display_df = display_df.rename(columns=existing_columns)
            
            # Formatação dos números
            numeric_columns = ['Confiança Média', 'Lift Médio', 'Confiança Máxima']
            for col in numeric_columns:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(3)
            
            # Exibir tabela
            st.dataframe(display_df, use_container_width=True)
            
            # Métricas de destaque
            st.markdown("### 🎯 Métricas de Destaque")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_rules = comparison_df['Rules_Found'].sum() if 'Rules_Found' in comparison_df.columns else 0
                st.metric(i18n.t('association.total_rules', 'Total de Regras'), f"{total_rules:,}")
            
            with col2:
                if 'Rules_Found' in comparison_df.columns and len(comparison_df) > 0:
                    best_algo = comparison_df.loc[comparison_df['Rules_Found'].idxmax(), 'Algorithm']
                    st.metric(i18n.t('association.best_algorithm', 'Melhor Algoritmo'), best_algo)
                else:
                    st.metric(i18n.t('association.best_algorithm', 'Melhor Algoritmo'), "N/A")
            
            with col3:
                if 'Avg_Confidence' in comparison_df.columns:
                    avg_confidence = comparison_df['Avg_Confidence'].mean()
                    st.metric(i18n.t('association.general_confidence', 'Confiança Geral'), f"{avg_confidence:.3f}")
                else:
                    st.metric(i18n.t('association.general_confidence', 'Confiança Geral'), "N/A")
            
            with col4:
                successful_algos = len(comparison_df[comparison_df['Execution_Status'] == 'Success']) if 'Execution_Status' in comparison_df.columns else 0
                st.metric(i18n.t('association.algorithms_ok', 'Algoritmos OK'), f"{successful_algos}/3")
            
            # Gráficos de comparação
            create_comparison_charts(comparison_df, i18n)
            
        except Exception as e:
            st.error(f"❌ Erro ao carregar comparação: {e}")
    else:
        st.info("📄 Arquivo de comparação não encontrado")

def create_comparison_charts(comparison_df, i18n):
    """Criar gráficos de comparação entre algoritmos"""
    
    if comparison_df.empty:
        return
    
    st.markdown("### 📈 Visualizações Comparativas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de barras - Número de regras
        if 'Rules_Found' in comparison_df.columns:
            fig_rules = px.bar(
                comparison_df, 
                x='Algorithm', 
                y='Rules_Found',
                title="📊 Número de Regras por Algoritmo",
                color='Rules_Found',
                color_continuous_scale='viridis',
                text='Rules_Found'
            )
            
            fig_rules.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig_rules.update_layout(
                title=dict(x=0.5, font=dict(size=16)),
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig_rules, use_container_width=True)
    
    with col2:
        # Gráfico radar - Métricas de qualidade
        if all(col in comparison_df.columns for col in ['Avg_Confidence', 'Avg_Lift', 'Max_Confidence']):
            
            fig_radar = go.Figure()
            
            for _, row in comparison_df.iterrows():
                if row['Execution_Status'] == 'Success':
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row['Avg_Confidence'], row['Avg_Lift']/2, row['Max_Confidence']],  # Normalizar lift
                        theta=['Confiança Média', 'Lift Médio', 'Confiança Máxima'],
                        fill='toself',
                        name=row['Algorithm'],
                        opacity=0.7
                    ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="🎯 Radar de Qualidade dos Algoritmos",
                title_x=0.5,
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)

def show_apriori_results(association_files, i18n):
    """Mostrar resultados específicos do Apriori"""
    st.subheader("🔍 " + i18n.t('association.apriori', 'Resultados do Algoritmo Apriori'))
    
    apriori_file = None
    for file in association_files:
        if 'apriori' in file.name.lower() and file.suffix == '.csv':
            apriori_file = file
            break
    
    if apriori_file:
        try:
            apriori_df = pd.read_csv(apriori_file)
            
            st.markdown(f"**📊 Total de regras encontradas:** {len(apriori_df):,}")
            
            # Filtros
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_confidence = st.slider(
                    i18n.t('association.min_confidence', 'Confiança Mínima'), 
                    0.0, 1.0, 0.6, 0.01,
                    key="apriori_conf"
                )
            
            with col2:
                min_lift = st.slider(
                    i18n.t('association.min_lift', 'Lift Mínimo'), 
                    0.0, 3.0, 1.0, 0.1,
                    key="apriori_lift"
                )
            
            with col3:
                top_n = st.selectbox(
                    i18n.t('association.top_rules', 'Top N regras'), 
                    [10, 20, 50, 100], 
                    index=1,
                    key="apriori_top"
                )
            
            # Filtrar dados
            if 'confidence' in apriori_df.columns and 'lift' in apriori_df.columns:
                filtered_df = apriori_df[
                    (apriori_df['confidence'] >= min_confidence) & 
                    (apriori_df['lift'] >= min_lift)
                ].head(top_n)
                
                if not filtered_df.empty:
                    st.markdown("### 🏆 Top Regras (Apriori)")
                    
                    # Formatação para exibição
                    display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
                    available_cols = [col for col in display_cols if col in filtered_df.columns]
                    
                    if available_cols:
                        display_df = filtered_df[available_cols].round(4)
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Gráfico de dispersão
                        fig_scatter = px.scatter(
                            filtered_df, 
                            x='confidence', 
                            y='lift',
                            size='support',
                            title="📈 Confiança vs Lift (Apriori)",
                            hover_data=['support']
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.warning("⚠️ Colunas esperadas não encontradas")
                else:
                    st.info("📄 Nenhuma regra atende aos critérios selecionados")
            else:
                st.dataframe(apriori_df.head(top_n), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Erro ao carregar resultados Apriori: {e}")
    else:
        st.info("📄 Resultados Apriori não encontrados")

def show_fp_growth_results(association_files, i18n):
    """Mostrar resultados específicos do FP-Growth"""
    st.subheader("🌳 " + i18n.t('association.fp_growth', 'Resultados do Algoritmo FP-Growth'))
    
    fp_growth_file = None
    for file in association_files:
        if 'fp_growth' in file.name.lower() and file.suffix == '.csv':
            fp_growth_file = file
            break
    
    if fp_growth_file:
        try:
            fp_growth_df = pd.read_csv(fp_growth_file)
            
            st.markdown(f"**📊 Total de regras encontradas:** {len(fp_growth_df):,}")
            
            # Interface similar ao Apriori
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_confidence = st.slider(
                    i18n.t('association.min_confidence', 'Confiança Mínima'), 
                    0.0, 1.0, 0.6, 0.01,
                    key="fp_growth_conf"
                )
            
            with col2:
                min_lift = st.slider(
                    i18n.t('association.min_lift', 'Lift Mínimo'), 
                    0.0, 3.0, 1.0, 0.1,
                    key="fp_growth_lift"
                )
            
            with col3:
                top_n = st.selectbox(
                    i18n.t('association.top_rules', 'Top N regras'), 
                    [10, 20, 50, 100], 
                    index=1,
                    key="fp_growth_top"
                )
            
            # Filtrar e mostrar resultados
            if 'confidence' in fp_growth_df.columns and 'lift' in fp_growth_df.columns:
                filtered_df = fp_growth_df[
                    (fp_growth_df['confidence'] >= min_confidence) & 
                    (fp_growth_df['lift'] >= min_lift)
                ].head(top_n)
                
                if not filtered_df.empty:
                    st.markdown("### 🏆 Top Regras (FP-Growth)")
                    
                    display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
                    available_cols = [col for col in display_cols if col in filtered_df.columns]
                    
                    if available_cols:
                        display_df = filtered_df[available_cols].round(4)
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Histograma de confiança
                        fig_hist = px.histogram(
                            filtered_df, 
                            x='confidence',
                            title="📊 Distribuição de Confiança (FP-Growth)",
                            nbins=20
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.info("📄 Nenhuma regra atende aos critérios")
            else:
                st.dataframe(fp_growth_df.head(top_n), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Erro ao carregar resultados FP-Growth: {e}")
    else:
        st.info("📄 Resultados FP-Growth não encontrados")

def show_eclat_results(association_files, i18n):
    """Mostrar resultados específicos do Eclat"""
    st.subheader("🔗 " + i18n.t('association.eclat', 'Resultados do Algoritmo Eclat'))
    
    eclat_file = None
    for file in association_files:
        if 'eclat' in file.name.lower() and file.suffix == '.csv':
            eclat_file = file
            break
    
    if eclat_file:
        try:
            eclat_df = pd.read_csv(eclat_file)
            
            st.markdown(f"**📊 Total de regras encontradas:** {len(eclat_df):,}")
            
            # Interface similar aos outros
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_confidence = st.slider(
                    i18n.t('association.min_confidence', 'Confiança Mínima'), 
                    0.0, 1.0, 0.6, 0.01,
                    key="eclat_conf"
                )
            
            with col2:
                min_lift = st.slider(
                    i18n.t('association.min_lift', 'Lift Mínimo'), 
                    0.0, 3.0, 1.0, 0.1,
                    key="eclat_lift"
                )
            
            with col3:
                top_n = st.selectbox(
                    i18n.t('association.top_rules', 'Top N regras'), 
                    [10, 20, 50, 100], 
                    index=1,
                    key="eclat_top"
                )
            
            # Filtrar e mostrar resultados
            if 'confidence' in eclat_df.columns and 'lift' in eclat_df.columns:
                filtered_df = eclat_df[
                    (eclat_df['confidence'] >= min_confidence) & 
                    (eclat_df['lift'] >= min_lift)
                ].head(top_n)
                
                if not filtered_df.empty:
                    st.markdown("### 🏆 Top Regras (Eclat)")
                    
                    display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
                    available_cols = [col for col in display_cols if col in filtered_df.columns]
                    
                    if available_cols:
                        display_df = filtered_df[available_cols].round(4)
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Box plot de lift
                        fig_box = px.box(
                            filtered_df, 
                            y='lift',
                            title="📊 Distribuição de Lift (Eclat)"
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("📄 Nenhuma regra atende aos critérios")
            else:
                st.dataframe(eclat_df.head(top_n), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Erro ao carregar resultados Eclat: {e}")
    else:
        st.info("📄 Resultados Eclat não encontrados")

def show_association_visualizations(association_files, i18n):
    """Mostrar visualizações de regras de associação"""
    st.subheader("📈 " + i18n.t('association.visualizations', 'Visualizações Avançadas'))
    
    # Combinar dados de todos os algoritmos para comparação
    all_rules = []
    algorithm_names = []
    
    for alg_name in ['apriori', 'fp_growth', 'eclat']:
        alg_file = None
        for file in association_files:
            if alg_name in file.name.lower() and file.suffix == '.csv':
                alg_file = file
                break
        
        if alg_file:
            try:
                df = pd.read_csv(alg_file)
                if not df.empty and 'confidence' in df.columns:
                    df['algorithm'] = alg_name.upper()
                    all_rules.append(df)
                    algorithm_names.append(alg_name.upper())
            except:
                continue
    
    if all_rules:
        combined_df = pd.concat(all_rules, ignore_index=True)
        
        # Gráfico de violino - Distribuição de métricas por algoritmo
        col1, col2 = st.columns(2)
        
        with col1:
            fig_violin_conf = px.violin(
                combined_df, 
                x='algorithm', 
                y='confidence',
                title="🎻 Distribuição de Confiança por Algoritmo",
                box=True
            )
            st.plotly_chart(fig_violin_conf, use_container_width=True)
        
        with col2:
            if 'lift' in combined_df.columns:
                fig_violin_lift = px.violin(
                    combined_df, 
                    x='algorithm', 
                    y='lift',
                    title="🎻 Distribuição de Lift por Algoritmo",
                    box=True
                )
                st.plotly_chart(fig_violin_lift, use_container_width=True)
        
        # Heatmap de correlação entre métricas
        if all(col in combined_df.columns for col in ['support', 'confidence', 'lift']):
            metrics_corr = combined_df[['support', 'confidence', 'lift']].corr()
            
            fig_heatmap = px.imshow(
                metrics_corr,
                title="🔥 Correlação entre Métricas",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Estatísticas resumidas
        st.markdown("### 📊 Estatísticas Resumidas")
        
        summary_stats = combined_df.groupby('algorithm').agg({
            'confidence': ['mean', 'std', 'max'],
            'lift': ['mean', 'std', 'max'] if 'lift' in combined_df.columns else lambda x: [0, 0, 0],
            'support': ['mean', 'std', 'max'] if 'support' in combined_df.columns else lambda x: [0, 0, 0]
        }).round(4)
        
        st.dataframe(summary_stats, use_container_width=True)
        
    else:
        st.info("📄 Nenhum arquivo de regras válido encontrado para visualizações")