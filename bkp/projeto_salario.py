# projeto_salario.py

# ================================================
# 1. IMPORTAÇÃO DE BIBLIOTECAS
# ================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

# ================================================
# 1.1. CONFIGURAÇÃO DE LOGGING
# ================================================
class EmojiFormatter(logging.Formatter):
    """Formatter personalizado com emojis"""
    
    emoji_mapping = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨',
        'SUCCESS': '✅'
    }
    
    def format(self, record):
        emoji = self.emoji_mapping.get(record.levelname, 'ℹ️')
        return f"{emoji} {record.getMessage()}"

# Aplicar o formatter personalizado
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(EmojiFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Remover handlers padrão para evitar duplicação
logger.handlers = [handler]

# ================================================
# 1.2. FUNÇÕES DE LOGGING AUXILIARES (DEFINIR PRIMEIRO)
# ================================================
def log_success(message):
    """Log com checkmark verde"""
    logging.info(f"✅ {message}")

def log_function_start(function_name):
    """Log início de função"""
    logging.info(f"🔄 Iniciando: {function_name}")

def log_function_end(function_name):
    """Log fim de função com sucesso"""
    logging.info(f"✅ Concluído: {function_name}")

def log_function(func):
    """Decorator para logging de início e fim de função"""
    def wrapper(*args, **kwargs):
        logging.info(f"🔄 Iniciando: {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"✅ Concluído: {func.__name__}")
        return result
    return wrapper

def get_memory_usage(df):
    """Calcular uso de memória do DataFrame"""
    memory_usage = df.memory_usage(deep=True).sum()
    return memory_usage / 1024 / 1024  # MB

# ================================================
# CONFIGURAÇÃO DE ESTILO MODERNO
# ================================================

# Configurar estilo moderno
plt.style.use('default')
sns.set_palette("husl")

# Configuração global CORRIGIDA
plt.rcParams.update({
    # Figura
    'figure.figsize': (12, 8),
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'none',
    
    # Eixos
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#dee2e6',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'axes.axisbelow': True,
    'axes.labelsize': 12,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.titlepad': 20,
    
    # Grid
    'grid.color': '#e9ecef',
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.7,
    
    # Texto - SEM FONTES PROBLEMÁTICAS
    'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 11,
    'text.color': '#343a40',
    
    # Legendas
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True,
    'legend.framealpha': 0.9,
    'legend.facecolor': 'white',
    'legend.edgecolor': '#dee2e6',
    
    # Ticks
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.color': '#6c757d',
    'ytick.color': '#6c757d',
    
    # Cores
    'patch.linewidth': 0.5,
    'patch.facecolor': '#007bff',
    'patch.edgecolor': '#0056b3',
    
    # Salvamento
    'savefig.dpi': 300,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
})

# Paleta de cores moderna
MODERN_COLORS = {
    'primary': '#007bff',
    'secondary': '#6c757d', 
    'success': '#28a745',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'gradient_blue': ['#667eea', '#764ba2'],
    'gradient_sunset': ['#ff9a9e', '#fecfef'],
    'gradient_ocean': ['#2196f3', '#21cbf3'],
    'categorical': ['#007bff', '#28a745', '#ffc107', '#dc3545', '#17a2b8', '#6f42c1', '#fd7e14']
}

# Função apply_modern_style CORRIGIDA
def apply_modern_style(ax, title="", subtitle="", remove_spines=True):
    """Aplicar estilo moderno consistente a um gráfico"""
    
    # Título principal (SEM EMOJIS PROBLEMÁTICOS)
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', 
                    color=MODERN_COLORS['dark'], pad=20)
    
    # Subtítulo
    if subtitle:
        ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, 
               fontsize=11, color=MODERN_COLORS['secondary'],
               ha='center', style='italic')
    
    # Remover spines desnecessários
    if remove_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(MODERN_COLORS['secondary'])
        ax.spines['bottom'].set_color(MODERN_COLORS['secondary'])
    
    # Grid sutil
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Cor de fundo
    ax.set_facecolor('#fafbfc')
    
    return ax

# Função save_modern_plot CORRIGIDA
def save_modern_plot(filename, dpi=300, transparent=False):
    """Salvar gráfico com configurações modernas - SEM EMOJIS"""
    try:
        plt.tight_layout()
        plt.savefig(f"imagens/{filename}", 
                    dpi=dpi, 
                    transparent=transparent,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none',
                    pad_inches=0.2)
        print(f"✓ Gráfico salvo: {filename}")  # Usar print simples
        plt.close()
    except Exception as e:
        print(f"✗ Erro ao salvar {filename}: {e}")
        plt.close()

# ================================================
# 2. CARREGAMENTO E LIMPEZA DOS DADOS
# ================================================
df = pd.read_csv('4-Carateristicas_salario.csv')
df = df.drop_duplicates()
logging.info("✅ Dados carregados e duplicatas removidas")

logging.info("\nResumo dos dados:")
logging.info(df.info())
logging.info("\nDescrição estatística:")
logging.info(df.describe())
logging.info("\nValores ausentes por coluna:")
logging.info(df.isnull().sum())

# ================================================
# 2.1. LIMPEZA E TIPAGEM DOS DADOS
# ================================================
logging.info("\n" + "="*60)
logging.info("LIMPEZA E TIPAGEM DOS DADOS")
logging.info("="*60)

def limpar_e_tipar_dados(df):
    """Limpar dados e aplicar tipagem correta às colunas"""
    log_function_start("Limpeza e tipagem de dados")
    
    # Fazer cópia para não alterar o original
    df_clean = df.copy()
    
    # Limpar espaços em branco em todas as colunas de texto
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    
    # Tratar valores '?' como NaN
    df_clean = df_clean.replace('?', pd.NA)
    
    logging.info("🔧 Aplicando tipagem correta às colunas...")
    
    # ===== TIPAGEM DAS VARIÁVEIS NUMÉRICAS =====
    numerical_columns_types = {
        'age': 'int16',           # Idade: 17-90 (int16 suficiente)
        'fnlwgt': 'int32',        # Peso final: pode ser grande (int32)
        'education-num': 'int8',  # Anos educação: 1-16 (int8 suficiente)
        'capital-gain': 'int32',  # Ganho capital: pode ser grande
        'capital-loss': 'int16',  # Perda capital: menor range
        'hours-per-week': 'int8'  # Horas semana: 1-99 (int8 suficiente)
    }
    
    for col, dtype in numerical_columns_types.items():
        if col in df_clean.columns:
            try:
                # Converter para numérico primeiro, depois para o tipo específico
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].astype(dtype)
                logging.info(f"✅ {col}: convertido para {dtype}")
            except Exception as e:
                logging.warning(f"⚠️ Erro ao converter {col}: {e}")
    
    # ===== TIPAGEM DAS VARIÁVEIS CATEGÓRICAS =====
    categorical_columns = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country', 'salary'
    ]
    
    for col in categorical_columns:
        if col in df_clean.columns:
            try:
                # Converter para categoria para economizar memória
                df_clean[col] = df_clean[col].astype('category')
                logging.info(f"✅ {col}: convertido para category")
            except Exception as e:
                logging.warning(f"⚠️ Erro ao converter {col}: {e}")
    
    # ===== VALIDAÇÃO DE RANGES =====
    logging.info("\n🔍 Validando ranges das variáveis...")
    
    # Validar idade
    if 'age' in df_clean.columns:
        invalid_age = (df_clean['age'] < 17) | (df_clean['age'] > 100)
        if invalid_age.any():
            logging.warning(f"⚠️ Encontradas {invalid_age.sum()} idades inválidas (fora de 17-100)")
            df_clean.loc[invalid_age, 'age'] = pd.NA

    # Validar anos de educação
    if 'education-num' in df_clean.columns:
        invalid_edu = (df_clean['education-num'] < 1) | (df_clean['education-num'] > 16)
        if invalid_edu.any():
            logging.warning(f"⚠️ Encontrados {invalid_edu.sum()} anos de educação inválidos (fora de 1-16)")
            df_clean.loc[invalid_edu, 'education-num'] = pd.NA

    # Validar horas por semana
    if 'hours-per-week' in df_clean.columns:
        invalid_hours = (df_clean['hours-per-week'] < 1) | (df_clean['hours-per-week'] > 99)
        if invalid_hours.any():
            logging.warning(f"⚠️ Encontradas {invalid_hours.sum()} horas/semana inválidas (fora de 1-99)")
            df_clean.loc[invalid_hours, 'hours-per-week'] = pd.NA

    # Validar ganhos/perdas de capital (não podem ser negativos)
    for col in ['capital-gain', 'capital-loss']:
        if col in df_clean.columns:
            invalid_capital = df_clean[col] < 0
            if invalid_capital.any():
                logging.warning(f"⚠️ Encontrados {invalid_capital.sum()} valores negativos em {col}")
                df_clean.loc[invalid_capital, col] = 0
    
    log_function_end("Limpeza e tipagem de dados")
    return df_clean, numerical_columns_types, categorical_columns

# Mostrar uso de memória antes da otimização
memory_before = get_memory_usage(df)
logging.info(f"💾 Uso de memória antes da otimização: {memory_before:.2f} MB")

# Aplicar limpeza e tipagem
df, numerical_columns_types, categorical_columns = limpar_e_tipar_dados(df)
logging.info("✅ Limpeza e tipagem de dados concluída")

# Mostrar uso de memória após otimização
memory_after = get_memory_usage(df)
logging.info(f"💾 Uso de memória após otimização: {memory_after:.2f} MB")
logging.info(f"📉 Redução: {((memory_before - memory_after) / memory_before * 100):.1f}%")

logging.info("\n✅ Dados limpos e tipados com sucesso!")
logging.info("\nInfo dos tipos de dados:")
logging.info(df.dtypes)

# Verificar valores ausentes após limpeza
missing_values = df.isnull().sum()
if missing_values.any():
    logging.warning(f"\n⚠️ Valores ausentes após limpeza:")
    logging.warning(missing_values[missing_values > 0])

# Agora podemos usar numerical_columns_types.keys() sem erro
numerical_columns = list(numerical_columns_types.keys())

# ================================================
# 2.2. REMOÇÃO DE OUTLIERS
# ================================================
from scipy.stats import zscore

@log_function
def remover_outliers(df, columns, threshold=3):
    """Remove outliers com base no Z-score"""
    df_filtered = df.copy()
    outliers_removed = 0
    
    for col in columns:
        if col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[col]):
            # Calcular Z-scores apenas para valores não nulos
            valid_mask = df_filtered[col].notna()
            if valid_mask.sum() > 0:  # Se há valores válidos
                z_scores = np.abs(zscore(df_filtered.loc[valid_mask, col]))
                outlier_mask = z_scores > threshold
                
                # Contar outliers antes de remover
                outliers_count = outlier_mask.sum()
                outliers_removed += outliers_count
                
                # Remover outliers
                outlier_indices = df_filtered.loc[valid_mask].index[outlier_mask]
                df_filtered = df_filtered.drop(outlier_indices)
                
                logging.info(f"🔍 {col}: {outliers_count} outliers removidos (Z-score > {threshold})")
    
    logging.info(f"📊 Total de registros removidos: {outliers_removed}")
    logging.info(f"📊 Registros restantes: {len(df_filtered)} de {len(df)} originais")
    
    return df_filtered

# Aplicar remoção de outliers
logging.info("\n" + "="*60)
logging.info("REMOÇÃO DE OUTLIERS")
logging.info("="*60)

# Verificar quais colunas numéricas existem no DataFrame
existing_numerical_cols = [col for col in numerical_columns if col in df.columns]
logging.info(f"Colunas numéricas encontradas: {existing_numerical_cols}")

if existing_numerical_cols:
    df_before_outliers = len(df)
    df = remover_outliers(df, existing_numerical_cols)
    df_after_outliers = len(df)
    
    reduction_percent = ((df_before_outliers - df_after_outliers) / df_before_outliers) * 100
    logging.info(f"📉 Redução do dataset: {reduction_percent:.1f}%")
    
    # Verificar se ainda temos dados suficientes
    if len(df) < 1000:
        logging.warning("⚠️ ATENÇÃO: Dataset muito pequeno após remoção de outliers!")
        logging.warning("Considere usar um threshold maior ou métodos alternativos.")
    else:
        logging.info("✅ Outliers removidos com sucesso")
else:
    logging.warning("⚠️ Nenhuma coluna numérica encontrada para remoção de outliers")

# ================================================
# FUNÇÕES DE VISUALIZAÇÃO MODERNA (DEFINIR ANTES DO USO)
# ================================================

def create_modern_histogram(data, column, title="", bins=30):
    """Criar histograma moderno com estatísticas - SEM EMOJIS PROBLEMÁTICOS"""
    
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
    
    # Curva de densidade suavizada
    try:
        from scipy import stats
        density = stats.gaussian_kde(valid_data)
        xs = np.linspace(valid_data.min(), valid_data.max(), 200)
        density_values = density(xs)
        
        # Escalar densidade para ajustar ao histograma
        density_scaled = density_values * len(valid_data) * (bins_edges[1] - bins_edges[0])
        
        ax2 = ax.twinx()
        ax2.plot(xs, density_scaled, color=MODERN_COLORS['danger'], 
                 linewidth=3, alpha=0.8, label='Densidade')
        ax2.set_ylabel('Densidade', color=MODERN_COLORS['danger'])
        ax2.tick_params(axis='y', labelcolor=MODERN_COLORS['danger'])
        ax2.spines['right'].set_color(MODERN_COLORS['danger'])
    except Exception as e:
        print(f"Aviso: Não foi possível adicionar curva de densidade para {column}: {e}")
    
    # Estatísticas no gráfico
    mean_val = valid_data.mean()
    median_val = valid_data.median()
    std_val = valid_data.std()
    
    # Linhas de referência
    ax.axvline(mean_val, color=MODERN_COLORS['success'], 
               linestyle='--', linewidth=2, alpha=0.8, label=f'Media: {mean_val:.1f}')
    ax.axvline(median_val, color=MODERN_COLORS['warning'], 
               linestyle='--', linewidth=2, alpha=0.8, label=f'Mediana: {median_val:.1f}')
    
    # Caixa de estatísticas (SEM EMOJIS PROBLEMÁTICOS)
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
    apply_modern_style(ax, title=title or f"Distribuicao de {column}")
    
    # Labels
    ax.set_xlabel(column.replace('-', ' ').replace('_', ' ').title(), 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequencia', fontsize=12, fontweight='bold')
    
    # Legenda
    ax.legend(loc='upper left', framealpha=0.9)
    
    return fig, ax

def create_modern_barplot(data, column, title="", top_n=15, horizontal=True):
    """Criar gráfico de barras moderno - SEM EMOJIS PROBLEMÁTICOS"""
    
    # Preparar dados
    value_counts = data[column].value_counts().head(top_n)
    
    if len(value_counts) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Sem dados para {column}', 
               ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    if horizontal:
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
        
    else:
        fig, ax = plt.subplots(figsize=(max(12, len(value_counts) * 0.8), 8))
        
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(value_counts)))
        
        bars = ax.bar(range(len(value_counts)), value_counts.values,
                     color=colors, edgecolor='white', linewidth=1)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, value_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + value * 0.01,
                   f'{value:,}', ha='center', va='bottom', fontweight='bold')
        
        # Configurar eixos
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Frequencia', fontsize=12, fontweight='bold')
    
    # Percentuais
    total = value_counts.sum()
    percentages = (value_counts / total * 100).round(1)
    
    # Caixa de informações (SEM EMOJIS PROBLEMÁTICOS)
    info_text = f"""Top {len(value_counts)} categorias:
    • Total registros: {total:,}
    • Categorias unicas: {data[column].nunique()}
    • Categoria mais comum: {value_counts.index[0]} ({percentages.iloc[0]:.1f}%)"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                    edgecolor=MODERN_COLORS['info'], alpha=0.9),
           verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    # Aplicar estilo moderno
    apply_modern_style(ax, title=title or f"Distribuicao de {column}")
    
    return fig, ax

def create_modern_correlation_matrix(data, title="Matriz de Correlacao"):
    """Criar matriz de correlação moderna - CORRIGIDA"""
    
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
    ax.set_title(title, fontsize=18, fontweight='bold', pad=30)
    
    # Rotacionar labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # CORREÇÃO: Usar gt() e lt() em vez de between()
    strong_corr_mask = correlation_matrix.abs() > 0.7
    moderate_corr_mask = (correlation_matrix.abs() > 0.3) & (correlation_matrix.abs() <= 0.7)
    
    strong_corr = strong_corr_mask.sum().sum() - len(correlation_matrix)  # Subtrair diagonal
    moderate_corr = moderate_corr_mask.sum().sum()
    
    # Caixa de informações (SEM EMOJIS PROBLEMÁTICOS)
    info_text = f"""Analise de Correlacao:
    • Correlacoes fortes (|r| > 0.7): {strong_corr}
    • Correlacoes moderadas (0.3 < |r| < 0.7): {moderate_corr}
    • Variaveis analisadas: {len(correlation_matrix)}"""
    
    ax.text(1.02, 0.98, info_text, transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                    edgecolor=MODERN_COLORS['primary'], alpha=0.9),
           verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    return fig, ax

def create_modern_feature_importance(importance_data, feature_names, title="Importancia das Features", top_n=20):
    """Criar gráfico moderno de importância das features - SEM EMOJIS PROBLEMÁTICOS"""
    
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
    
    # Caixa de informações (SEM EMOJIS PROBLEMÁTICOS)
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
    apply_modern_style(ax, title=title)
    
    return fig, ax

def create_modern_confusion_matrix(y_true, y_pred, model_name="Modelo"):
    """Criar matriz de confusão moderna - SEM EMOJIS PROBLEMÁTICOS"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Heatmap da matriz de confusão
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 14, "weight": "bold"},
                ax=ax)
    
    # Labels
    ax.set_xlabel('Predicao', fontsize=12, fontweight='bold')
    ax.set_ylabel('Real', fontsize=12, fontweight='bold')
    ax.set_title(f'Matriz de Confusao - {model_name}', fontsize=16, fontweight='bold', pad=20)
    
    # Configurar ticks
    ax.set_xticklabels(['≤ 50K', '> 50K'])
    ax.set_yticklabels(['≤ 50K', '> 50K'])
    
    # Calcular métricas
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, cm[1,1])
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Caixa de métricas (SEM EMOJIS PROBLEMÁTICOS)
    metrics_text = f"""Metricas:
    • Accuracy: {accuracy:.3f}
    • Precision: {precision:.3f}
    • Recall: {recall:.3f}
    • F1-Score: {f1:.3f}
    
    Matriz:
    • VP: {tp}  |  FP: {fp}
    • FN: {fn}  |  VN: {tn}"""
    
    ax.text(1.05, 0.5, metrics_text, transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                    edgecolor=MODERN_COLORS['primary'], alpha=0.9),
           verticalalignment='center', fontsize=10, fontfamily='monospace')
    
    return fig, ax

# ================================================
# 3. ANÁLISE EXPLORATÓRIA DE DADOS (EDA) - VERSÃO CORRIGIDA
# ================================================
logging.info("\n" + "="*60)
logging.info("GERANDO VISUALIZAÇÕES MODERNAS")
logging.info("="*60)

# Criar diretório para imagens se não existir
os.makedirs("imagens", exist_ok=True)

try:
    # 1. Histogramas modernos para variáveis numéricas
    logging.info("📊 Gerando histogramas modernos...")
    numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    for col in numerical_cols:
        if col in df.columns:
            logging.info(f"  📈 Processando {col}")
            try:
                fig, ax = create_modern_histogram(df, col, title=f"Distribuicao de {col.replace('-', ' ').title()}")
                save_modern_plot(f"hist_{col}.png")
            except Exception as e:
                logging.warning(f"Erro ao gerar histograma para {col}: {e}")

    # 2. Gráficos categóricos modernos
    logging.info("📊 Gerando gráficos categóricos modernos...")
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                        'relationship', 'race', 'sex', 'native-country']

    for col in categorical_cols:
        if col in df.columns and df[col].nunique() < 50:
            logging.info(f"  📈 Processando {col}")
            try:
                fig, ax = create_modern_barplot(df, col, title=f"Distribuicao de {col.replace('-', ' ').title()}")
                save_modern_plot(f"{col}_distribution.png")
            except Exception as e:
                logging.warning(f"Erro ao gerar gráfico para {col}: {e}")

    # 3. Distribuição da variável target
    logging.info("📊 Gerando distribuição da variável target...")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        salary_counts = df['salary'].value_counts()
        
        if len(salary_counts) > 1:
            colors = [MODERN_COLORS['success'], MODERN_COLORS['danger']]
            wedges, texts, autotexts = ax.pie(salary_counts.values, 
                                             labels=['≤ 50K', '> 50K'] if 0 in salary_counts.index else salary_counts.index,
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

        save_modern_plot("salary_distribution.png")
    except Exception as e:
        logging.warning(f"Erro ao gerar distribuição de salários: {e}")

    # 4. Matriz de correlação moderna
    logging.info("📊 Gerando matriz de correlação moderna...")
    try:
        fig, ax = create_modern_correlation_matrix(df, title="Matriz de Correlacao - Variaveis Numericas")
        save_modern_plot("correlacao.png")
    except Exception as e:
        logging.warning(f"Erro ao gerar matriz de correlação: {e}")

    logging.info("✅ Todas as visualizações modernas foram geradas!")

except Exception as e:
    logging.error(f"Erro geral na geração de visualizações: {e}")