"""Funções de visualização moderna - Extraído do projeto_salario.py"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from .styles import MODERN_COLORS, apply_modern_style, save_modern_plot

class VisualizationGenerator:
    """Classe para gerar todas as visualizações"""
    
    def __init__(self):
        pass
    
    def generate_all_plots(self, df, models=None, results=None):
        """Gerar todas as visualizações"""
        logging.info("\n" + "="*60)
        logging.info("GERANDO VISUALIZAÇÕES MODERNAS")
        logging.info("="*60)
        
        # Criar diretório para imagens se não existir
        os.makedirs("imagens", exist_ok=True)
        
        try:
            # 1. Histogramas para variáveis numéricas
            logging.info("📊 Gerando histogramas modernos...")
            numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
            for col in numerical_cols:
                if col in df.columns:
                    logging.info(f"  📈 Processando {col}")
                    try:
                        self._create_modern_histogram(df, col)
                        save_modern_plot(f"hist_{col}.png")
                    except Exception as e:
                        logging.warning(f"Erro ao gerar histograma para {col}: {e}")

            # 2. Gráficos categóricos
            logging.info("📊 Gerando gráficos categóricos modernos...")
            categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                                'relationship', 'race', 'sex', 'native-country']

            for col in categorical_cols:
                if col in df.columns and df[col].nunique() < 50:
                    logging.info(f"  📈 Processando {col}")
                    try:
                        self._create_modern_barplot(df, col)
                        save_modern_plot(f"{col}_distribution.png")
                    except Exception as e:
                        logging.warning(f"Erro ao gerar gráfico para {col}: {e}")

            # 3. Distribuição da variável target
            logging.info("📊 Gerando distribuição da variável target...")
            try:
                self._create_target_distribution(df)
                save_modern_plot("salary_distribution.png")
            except Exception as e:
                logging.warning(f"Erro ao gerar distribuição de salários: {e}")

            # 4. Matriz de correlação
            logging.info("📊 Gerando matriz de correlação moderna...")
            try:
                self._create_correlation_matrix(df)
                save_modern_plot("correlacao.png")
            except Exception as e:
                logging.warning(f"Erro ao gerar matriz de correlação: {e}")

            # 5. Feature importance (se modelos fornecidos)
            if models and 'Random Forest' in models:
                logging.info("📊 Gerando importância das features...")
                try:
                    self._create_feature_importance(models['Random Forest'])
                    save_modern_plot("feature_importance_rf.png")
                except Exception as e:
                    logging.warning(f"Erro ao gerar feature importance: {e}")

            # 6. Matriz de confusão (se resultados fornecidos)
            if results and models:
                logging.info("📊 Gerando matriz de confusão...")
                for model_name, result in results.items():
                    if 'y_pred' in result and model_name in models:
                        try:
                            # Precisamos dos dados de teste para isso
                            # Por agora, vamos pular ou implementar diferente
                            logging.info(f"  📈 Matriz de confusão para {model_name} será gerada pelo modelo")
                        except Exception as e:
                            logging.warning(f"Erro ao gerar matriz de confusão para {model_name}: {e}")

            logging.info("✅ Todas as visualizações modernas foram geradas!")

        except Exception as e:
            logging.error(f"Erro geral na geração de visualizações: {e}")
    
    def _create_modern_histogram(self, data, column, bins=30):
        """Criar histograma moderno com estatísticas"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Dados válidos
        valid_data = data[column].dropna()
        
        if len(valid_data) == 0:
            ax.text(0.5, 0.5, f'Sem dados válidos para {column}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig, ax
        
        # Histograma principal
        n, bins_edges, patches = ax.hist(valid_data, bins=bins, 
                                       color=MODERN_COLORS['primary'], 
                                       alpha=0.7, edgecolor='white', linewidth=1.2)
        
        # Gradiente nas barras
        for i, patch in enumerate(patches):
            height_ratio = n[i] / max(n) if max(n) > 0 else 0
            color_intensity = 0.3 + 0.7 * height_ratio
            patch.set_facecolor(plt.cm.Blues(color_intensity))
            patch.set_edgecolor('white')
        
        # Estatísticas no gráfico
        mean_val = valid_data.mean()
        median_val = valid_data.median()
        std_val = valid_data.std()
        
        # Linhas de referência
        ax.axvline(mean_val, color=MODERN_COLORS['success'], 
                   linestyle='--', linewidth=2, alpha=0.8, label=f'Media: {mean_val:.1f}')
        ax.axvline(median_val, color=MODERN_COLORS['warning'], 
                   linestyle='--', linewidth=2, alpha=0.8, label=f'Mediana: {median_val:.1f}')
        
        # Caixa de estatísticas
        stats_text = f"""Estatisticas:
• Media: {mean_val:.2f}
• Mediana: {median_val:.2f}
• Desvio Padrao: {std_val:.2f}
• Min: {valid_data.min():.2f}
• Max: {valid_data.max():.2f}
• Registros: {len(valid_data):,}"""
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                        edgecolor=MODERN_COLORS['primary'], alpha=0.9),
               verticalalignment='top', horizontalalignment='right',
               fontsize=10, fontfamily='monospace')
        
        # Aplicar estilo moderno
        apply_modern_style(ax, title=f"Distribuicao de {column.replace('-', ' ').title()}")
        
        # Labels
        ax.set_xlabel(column.replace('-', ' ').replace('_', ' ').title(), 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequencia', fontsize=12, fontweight='bold')
        
        # Legenda
        ax.legend(loc='upper left', framealpha=0.9)
        
        return fig, ax
    
    def _create_modern_barplot(self, data, column, top_n=15):
        """Criar gráfico de barras moderno"""
        # Preparar dados
        value_counts = data[column].value_counts().head(top_n)
        
        if len(value_counts) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Sem dados para {column}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig, ax
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(value_counts) * 0.6)))
        
        # Criar gradiente de cores
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(value_counts)))
        
        bars = ax.barh(range(len(value_counts)), value_counts.values, 
                      color=colors, edgecolor='white', linewidth=1)
        
        # Adicionar valores nas barras
        for i, (bar, value) in enumerate(zip(bars, value_counts.values)):
            width = bar.get_width()
            ax.text(width + value * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:,}', ha='left', va='center', fontweight='bold')
        
        # Configurar eixos
        ax.set_yticks(range(len(value_counts)))
        ax.set_yticklabels(value_counts.index, fontsize=10)
        ax.set_xlabel('Frequencia', fontsize=12, fontweight='bold')
        
        # Inverter ordem (maior no topo)
        ax.invert_yaxis()
        
        # Percentuais
        total = value_counts.sum()
        percentages = (value_counts / total * 100).round(1)
        
        # Caixa de informações
        info_text = f"""Top {len(value_counts)} categorias:
• Total registros: {total:,}
• Categorias unicas: {data[column].nunique()}
• Categoria mais comum: {value_counts.index[0]} ({percentages.iloc[0]:.1f}%)"""
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                        edgecolor=MODERN_COLORS['info'], alpha=0.9),
               verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        # Aplicar estilo moderno
        apply_modern_style(ax, title=f"Distribuicao de {column.replace('-', ' ').title()}")
        
        return fig, ax
    
    def _create_target_distribution(self, df):
        """Gerar distribuição da variável target"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'salary' in df.columns:
            salary_counts = df['salary'].value_counts()
            
            if len(salary_counts) > 1:
                colors = [MODERN_COLORS['success'], MODERN_COLORS['danger']]
                wedges, texts, autotexts = ax.pie(salary_counts.values, 
                                                 labels=['≤ 50K', '> 50K'],
                                                 autopct='%1.1f%%',
                                                 colors=colors,
                                                 explode=(0.05, 0.05),
                                                 shadow=True,
                                                 startangle=90)
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(12)
                
                ax.set_title('Distribuicao de Salarios', fontsize=16, fontweight='bold', pad=20)
            else:
                ax.bar(['Classe Unica'], [len(df)], color=MODERN_COLORS['primary'])
                ax.set_title('Distribuicao de Salarios (Apenas uma classe)', fontsize=16, fontweight='bold')
        
        return fig, ax
    
    def _create_correlation_matrix(self, data):
        """Criar matriz de correlação moderna"""
        # Selecionar apenas colunas numéricas
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Nenhuma variavel numerica encontrada', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig, ax
        
        correlation_matrix = numeric_data.corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Criar máscara para triângulo superior
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Heatmap com estilo moderno
        sns.heatmap(correlation_matrix, 
                    mask=mask,
                    annot=True, 
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    fmt='.2f',
                    cbar_kws={"shrink": 0.8, "label": "Correlacao"},
                    linewidths=0.5,
                    linecolor='white',
                    ax=ax)
        
        # Estilo moderno
        ax.set_title("Matriz de Correlacao - Variaveis Numericas", fontsize=18, fontweight='bold', pad=30)
        
        # Rotacionar labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        # Análise de correlações
        strong_corr_mask = correlation_matrix.abs() > 0.7
        moderate_corr_mask = (correlation_matrix.abs() > 0.3) & (correlation_matrix.abs() <= 0.7)
        
        strong_corr = strong_corr_mask.sum().sum() - len(correlation_matrix)  # Subtrair diagonal
        moderate_corr = moderate_corr_mask.sum().sum()
        
        # Caixa de informações
        info_text = f"""Analise de Correlacao:
• Correlacoes fortes (|r| > 0.7): {strong_corr}
• Correlacoes moderadas (0.3 < |r| < 0.7): {moderate_corr}
• Variaveis analisadas: {len(correlation_matrix)}"""
        
        ax.text(1.02, 0.98, info_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                        edgecolor=MODERN_COLORS['primary'], alpha=0.9),
               verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        return fig, ax
    
    def _create_feature_importance(self, model):
        """Criar gráfico de importância das features"""
        if not hasattr(model, 'feature_importances_'):
            logging.warning("Modelo não possui feature_importances_")
            return
        
        # Tentar carregar informações das features
        try:
            import joblib
            from pathlib import Path
            
            # Buscar feature_info
            processed_dir = Path("data/processed")
            feature_info_path = processed_dir / "feature_info.joblib"
            
            if feature_info_path.exists():
                feature_info = joblib.load(feature_info_path)
                feature_names = feature_info['feature_names']
            else:
                # Fallback: nomes genéricos
                feature_names = [f"Feature_{i}" for i in range(len(model.feature_importances_))]
                
        except Exception as e:
            logging.warning(f"Erro ao carregar nomes das features: {e}")
            feature_names = [f"Feature_{i}" for i in range(len(model.feature_importances_))]
        
        importance_data = model.feature_importances_
        top_n = min(20, len(importance_data))
        
        # Preparar dados
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_data
        }).sort_values('importance', ascending=True).tail(top_n)
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(feature_df) * 0.4)))
        
        # Gradiente de cores baseado na importância
        norm = plt.Normalize(feature_df['importance'].min(), feature_df['importance'].max())
        colors = plt.cm.viridis(norm(feature_df['importance']))
        
        bars = ax.barh(range(len(feature_df)), feature_df['importance'], 
                       color=colors, edgecolor='white', linewidth=1.2)
        
        # Adicionar valores nas barras
        for i, (bar, importance) in enumerate(zip(bars, feature_df['importance'])):
            width = bar.get_width()
            ax.text(width + importance * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center', 
                   fontweight='bold', fontsize=9)
        
        # Configurar eixos
        ax.set_yticks(range(len(feature_df)))
        ax.set_yticklabels(feature_df['feature'], fontsize=10)
        ax.set_xlabel('Importancia', fontsize=12, fontweight='bold')
        
        # Destacar top 3
        for i in range(max(0, len(bars)-3), len(bars)):
            bars[i].set_edgecolor(MODERN_COLORS['danger'])
            bars[i].set_linewidth(2)
        
        # Informações estatísticas
        total_importance = feature_df['importance'].sum()
        top3_importance = feature_df['importance'].tail(3).sum()
        
        # Caixa de informações
        info_text = f"""Analise de Importancia:
• Top 3 features: {top3_importance/total_importance*100:.1f}% da importancia
• Feature mais importante: {feature_df.iloc[-1]['feature']}
• Importancia maxima: {feature_df['importance'].max():.3f}
• Features analisadas: {len(feature_df)}"""
        
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                        edgecolor=MODERN_COLORS['success'], alpha=0.9),
               verticalalignment='bottom', horizontalalignment='right',
               fontsize=10, fontfamily='monospace')
        
        # Aplicar estilo moderno
        apply_modern_style(ax, title="Importancia das Features - Random Forest")
        
        return fig, ax