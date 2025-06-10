"""
📊 Componente de Gráficos Multilingual Avançado
Visualizações modernas com suporte completo a traduções
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class ChartComponents:
    """Sistema avançado de gráficos multilinguals"""
    
    def __init__(self, i18n_system):
        self.i18n = i18n_system
        
        # Configurações padrão dos gráficos
        self.default_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'chart',
                'height': 600,
                'width': 800,
                'scale': 2
            }
        }
        
        # Paleta de cores moderna
        self.color_palettes = {
            'default': px.colors.qualitative.Set3,
            'professional': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'modern': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'],
            'academic': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
        }
    
    def create_modern_pie_chart(self, 
                               df: pd.DataFrame, 
                               values: List, 
                               names: List, 
                               title_key: str,
                               color_palette: str = 'modern') -> go.Figure:
        """Criar gráfico de pizza moderno"""
        try:
            # Traduzir título
            title = self.i18n.t(title_key, title_key.replace('_', ' ').title())
            
            # Traduzir nomes se necessário
            translated_names = [self.i18n.translate_data_value(name) for name in names]
            
            # Criar gráfico
            fig = px.pie(
                values=values,
                names=translated_names,
                title=f"📊 {title}",
                color_discrete_sequence=self.color_palettes.get(color_palette, self.color_palettes['modern'])
            )
            
            # Personalizar layout
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Valor: %{value}<br>Percentual: %{percent}<extra></extra>',
                hole=0.3  # Donut style
            )
            
            fig.update_layout(
                title=dict(x=0.5, font=dict(size=18)),
                font=dict(size=12),
                plot_bgcolor='rgba(248,249,250,0.8)',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar gráfico de pizza: {e}")
            return self._create_error_chart(str(e))
    
    def create_modern_histogram(self, 
                               df: pd.DataFrame, 
                               column: str, 
                               title_key: str,
                               color_column: Optional[str] = None,
                               bins: int = 30) -> go.Figure:
        """Criar histograma moderno"""
        try:
            # Verificar se coluna existe
            if column not in df.columns:
                logger.warning(f"⚠️ Coluna {column} não encontrada")
                return None
            
            # Traduzir título
            title = self.i18n.t(title_key, title_key.replace('_', ' ').title())
            
            # Criar histograma
            if color_column and color_column in df.columns:
                # Histograma colorido por categoria
                fig = px.histogram(
                    df, 
                    x=column, 
                    color=color_column,
                    title=f"📊 {title}",
                    nbins=bins,
                    opacity=0.7,
                    color_discrete_sequence=self.color_palettes['modern']
                )
            else:
                # Histograma simples
                fig = px.histogram(
                    df, 
                    x=column, 
                    title=f"📊 {title}",
                    nbins=bins,
                    color_discrete_sequence=['#667eea']
                )
            
            # Personalizar layout
            fig.update_layout(
                title=dict(x=0.5, font=dict(size=18)),
                xaxis_title=self.i18n.translate_data_value(column),
                yaxis_title=self.i18n.t('charts.frequency', 'Frequência'),
                plot_bgcolor='rgba(248,249,250,0.8)',
                bargap=0.1
            )
            
            # Adicionar estatísticas
            mean_val = df[column].mean()
            median_val = df[column].median()
            
            fig.add_vline(
                x=mean_val, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Média: {mean_val:.2f}"
            )
            
            fig.add_vline(
                x=median_val, 
                line_dash="dash", 
                line_color="blue",
                annotation_text=f"Mediana: {median_val:.2f}"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar histograma: {e}")
            return self._create_error_chart(str(e))
    
    def create_interactive_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Criar heatmap de correlação interativo"""
        try:
            # Selecionar apenas colunas numéricas
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                logger.warning("⚠️ Nenhuma coluna numérica encontrada")
                return None
            
            # Calcular correlação
            corr_matrix = numeric_df.corr()
            
            # Criar heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate='<b>%{x} vs %{y}</b><br>Correlação: %{z:.3f}<extra></extra>'
            ))
            
            # Personalizar layout
            fig.update_layout(
                title=dict(
                    text=f"🔗 {self.i18n.t('charts.correlation_matrix', 'Matriz de Correlação')}",
                    x=0.5,
                    font=dict(size=18)
                ),
                xaxis=dict(side='bottom'),
                yaxis=dict(side='left'),
                plot_bgcolor='rgba(248,249,250,0.8)',
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar heatmap: {e}")
            return self._create_error_chart(str(e))
    
    def create_advanced_scatter_plot(self, 
                                   df: pd.DataFrame,
                                   x_col: str,
                                   y_col: str,
                                   color_col: Optional[str] = None,
                                   size_col: Optional[str] = None,
                                   title: Optional[str] = None) -> go.Figure:
        """Criar scatter plot avançado"""
        try:
            # Verificar colunas
            required_cols = [x_col, y_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"⚠️ Colunas não encontradas: {missing_cols}")
                return None
            
            # Título padrão
            if not title:
                title = f"📈 {x_col} vs {y_col}"
            
            # Criar scatter plot
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                size=size_col,
                title=title,
                opacity=0.7,
                color_discrete_sequence=self.color_palettes['modern'],
                hover_data={col: True for col in df.columns if col in [x_col, y_col, color_col, size_col]}
            )
            
            # Adicionar linha de tendência
            if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
                # Calcular correlação
                correlation = df[x_col].corr(df[y_col])
                
                # Adicionar linha de tendência
                z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
                p = np.poly1d(z)
                
                fig.add_scatter(
                    x=df[x_col],
                    y=p(df[x_col]),
                    mode='lines',
                    name=f'Tendência (r={correlation:.3f})',
                    line=dict(dash='dash', color='red')
                )
            
            # Personalizar layout
            fig.update_layout(
                title=dict(x=0.5, font=dict(size=18)),
                xaxis_title=self.i18n.translate_data_value(x_col),
                yaxis_title=self.i18n.translate_data_value(y_col),
                plot_bgcolor='rgba(248,249,250,0.8)',
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar scatter plot: {e}")
            return self._create_error_chart(str(e))
    
    def create_dashboard_metrics_chart(self, metrics: Dict[str, Any]) -> go.Figure:
        """Criar gráfico de métricas do dashboard"""
        try:
            # Preparar dados
            categories = list(metrics.keys())
            values = list(metrics.values())
            
            # Traduzir categorias
            translated_categories = [self.i18n.translate_data_value(cat) for cat in categories]
            
            # Criar gráfico de barras horizontal
            fig = go.Figure(data=[
                go.Bar(
                    y=translated_categories,
                    x=values,
                    orientation='h',
                    marker=dict(
                        color=self.color_palettes['modern'][:len(categories)],
                        line=dict(color='rgba(50,50,50,0.8)', width=1)
                    ),
                    text=values,
                    textposition='outside'
                )
            ])
            
            # Personalizar layout
            fig.update_layout(
                title=dict(
                    text=f"📊 {self.i18n.t('charts.dashboard_metrics', 'Métricas do Dashboard')}",
                    x=0.5,
                    font=dict(size=18)
                ),
                xaxis_title=self.i18n.t('charts.values', 'Valores'),
                plot_bgcolor='rgba(248,249,250,0.8)',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar gráfico de métricas: {e}")
            return self._create_error_chart(str(e))
    
    def create_time_series_chart(self, 
                               df: pd.DataFrame,
                               date_col: str,
                               value_col: str,
                               title_key: str) -> go.Figure:
        """Criar gráfico de série temporal"""
        try:
            # Verificar colunas
            if date_col not in df.columns or value_col not in df.columns:
                logger.warning(f"⚠️ Colunas não encontradas: {date_col}, {value_col}")
                return None
            
            # Traduzir título
            title = self.i18n.t(title_key, title_key.replace('_', ' ').title())
            
            # Criar gráfico de linha
            fig = px.line(
                df,
                x=date_col,
                y=value_col,
                title=f"📈 {title}",
                markers=True
            )
            
            # Personalizar layout
            fig.update_traces(
                line=dict(color='#667eea', width=3),
                marker=dict(size=8, color='#764ba2')
            )
            
            fig.update_layout(
                title=dict(x=0.5, font=dict(size=18)),
                xaxis_title=self.i18n.translate_data_value(date_col),
                yaxis_title=self.i18n.translate_data_value(value_col),
                plot_bgcolor='rgba(248,249,250,0.8)',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar série temporal: {e}")
            return self._create_error_chart(str(e))
    
    def create_metric_card(self, title: str, value: str, icon: str = "📊") -> str:
        """Criar card de métrica HTML"""
        return f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">
                {icon}
            </div>
            <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">
                {title}
            </div>
            <div style="font-size: 2rem; font-weight: bold;">
                {value}
            </div>
        </div>
        """
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Criar gráfico de erro"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"❌ Erro: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            title="Erro na Visualização",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='rgba(248,249,250,0.8)'
        )
        
        return fig
    
    def get_chart_config(self, custom_config: Optional[Dict] = None) -> Dict:
        """Obter configuração dos gráficos"""
        config = self.default_config.copy()
        
        if custom_config:
            config.update(custom_config)
        
        return config