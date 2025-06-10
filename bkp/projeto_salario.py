"""
Sistema Completo de Análise Salarial - Versão Académica Final
Projeto de Data Science com análises supervisionadas e não supervisionadas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from datetime import datetime
import warnings

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore')

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('salary_analysis.log', encoding='utf-8')
    ]
)

# =============================================================================
# 1. PROCESSAMENTO DE DADOS
# =============================================================================

class DataProcessor:
    """Processador completo de dados salariais"""
    
    def __init__(self):
        self.df = None
        self.processed_df = None
        
    def load_data(self, file_path='4-Carateristicas_salario.csv'):
        """Carregar dados do CSV"""
        try:
            if Path(file_path).exists():
                self.df = pd.read_csv(file_path)
            elif Path(f"data/raw/{file_path}").exists():
                self.df = pd.read_csv(f"data/raw/{file_path}")
            else:
                raise FileNotFoundError(f"Arquivo {file_path} não encontrado")
            
            logging.info(f"✅ Dados carregados: {self.df.shape}")
            return self.df
        except Exception as e:
            logging.error(f"❌ Erro ao carregar dados: {e}")
            raise
    
    def clean_data(self):
        """Limpeza e preprocessamento dos dados"""
        if self.df is None:
            raise ValueError("Carregue os dados primeiro com load_data()")
        
        logging.info("🧹 Iniciando limpeza dos dados...")
        
        # Remover espaços em branco
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].astype(str).str.strip()
        
        # Substituir '?' por NaN
        self.df = self.df.replace('?', np.nan)
        
        # Remover linhas com muitos valores ausentes
        missing_threshold = 0.5
        self.df = self.df.dropna(thresh=len(self.df.columns) * missing_threshold)
        
        # Tratar valores ausentes específicos
        if 'workclass' in self.df.columns:
            self.df['workclass'].fillna('Unknown', inplace=True)
        if 'occupation' in self.df.columns:
            self.df['occupation'].fillna('Unknown', inplace=True)
        if 'native-country' in self.df.columns:
            self.df['native-country'].fillna('United-States', inplace=True)
        
        logging.info(f"✅ Dados limpos: {self.df.shape}")
        return self.df
    
    def create_features(self):
        """Criar novas features"""
        logging.info("🔧 Criando novas features...")
        
        # Feature de faixa etária
        if 'age' in self.df.columns:
            self.df['age_group'] = pd.cut(self.df['age'], 
                                        bins=[0, 25, 35, 45, 55, 100],
                                        labels=['muito_jovem', 'jovem', 'adulto', 'maduro', 'senior'])
        
        # Feature de horas trabalhadas
        if 'hours-per-week' in self.df.columns:
            self.df['work_intensity'] = pd.cut(self.df['hours-per-week'],
                                             bins=[0, 35, 45, 60, 100],
                                             labels=['part_time', 'normal', 'overtime', 'workaholic'])
        
        # Feature de ganho/perda de capital
        if 'capital-gain' in self.df.columns and 'capital-loss' in self.df.columns:
            self.df['net_capital'] = self.df['capital-gain'] - self.df['capital-loss']
            self.df['has_capital_gain'] = (self.df['capital-gain'] > 0).astype(int)
            self.df['has_capital_loss'] = (self.df['capital-loss'] > 0).astype(int)
        
        # Feature de educação combinada
        if 'education' in self.df.columns and 'education-num' in self.df.columns:
            self.df['education_level'] = pd.cut(self.df['education-num'],
                                              bins=[0, 9, 12, 16, 20],
                                              labels=['basico', 'medio', 'superior', 'pos_graduacao'])
        
        logging.info("✅ Novas features criadas")
        return self.df

# =============================================================================
# 2. ANÁLISE EXPLORATÓRIA
# =============================================================================

class ExploratoryAnalysis:
    """Análise exploratória completa dos dados"""
    
    def __init__(self, df):
        self.df = df
        self.setup_style()
    
    def setup_style(self):
        """Configurar estilo dos gráficos"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def generate_basic_statistics(self):
        """Estatísticas descritivas básicas"""
        logging.info("📊 Gerando estatísticas descritivas...")
        
        # Criar diretório de saída
        output_dir = Path("output/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Estatísticas numéricas
        numeric_stats = self.df.describe()
        numeric_stats.to_csv(output_dir / "numeric_statistics.csv")
        
        # Estatísticas categóricas
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        categorical_stats = {}
        
        for col in categorical_cols:
            categorical_stats[col] = {
                'unique_values': self.df[col].nunique(),
                'most_frequent': self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'N/A',
                'missing_values': self.df[col].isnull().sum()
            }
        
        categorical_df = pd.DataFrame(categorical_stats).T
        categorical_df.to_csv(output_dir / "categorical_statistics.csv")
        
        # Distribuição da variável target
        if 'salary' in self.df.columns:
            salary_dist = self.df['salary'].value_counts(normalize=True)
            logging.info(f"📈 Distribuição salarial: {salary_dist.to_dict()}")
        
        logging.info("✅ Estatísticas básicas geradas")
    
    def create_visualizations(self):
        """Criar visualizações exploratórias"""
        logging.info("🎨 Gerando visualizações exploratórias...")
        
        # Criar diretório de imagens
        images_dir = Path("output/images")
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Distribuição da variável target
        if 'salary' in self.df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Distribuição de salário
            self.df['salary'].value_counts().plot(kind='bar', ax=axes[0,0])
            axes[0,0].set_title('Distribuição de Salário')
            axes[0,0].set_ylabel('Frequência')
            
            # Salário por idade
            if 'age' in self.df.columns:
                sns.boxplot(data=self.df, x='salary', y='age', ax=axes[0,1])
                axes[0,1].set_title('Distribuição de Idade por Salário')
            
            # Salário por sexo
            if 'sex' in self.df.columns:
                salary_sex = pd.crosstab(self.df['sex'], self.df['salary'], normalize='index')
                salary_sex.plot(kind='bar', ax=axes[1,0])
                axes[1,0].set_title('Distribuição de Salário por Sexo')
                axes[1,0].legend(title='Salário')
            
            # Salário por educação
            if 'education' in self.df.columns:
                top_education = self.df['education'].value_counts().head(8).index
                education_salary = self.df[self.df['education'].isin(top_education)]
                salary_edu = pd.crosstab(education_salary['education'], education_salary['salary'], normalize='index')
                salary_edu.plot(kind='bar', ax=axes[1,1])
                axes[1,1].set_title('Distribuição de Salário por Educação (Top 8)')
                axes[1,1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(images_dir / "exploratory_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Matriz de correlação
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Matriz de Correlação - Variáveis Numéricas')
            plt.tight_layout()
            plt.savefig(images_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logging.info("✅ Visualizações exploratórias criadas")

# =============================================================================
# 3. MODELAGEM DE MACHINE LEARNING
# =============================================================================

class ModelTrainer:
    """Treinamento e avaliação de modelos de ML"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.preprocessor = None
    
    def prepare_data(self, df):
        """Preparar dados para modelagem"""
        logging.info("🔧 Preparando dados para modelagem...")
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        
        # Separar features e target
        X = df.drop('salary', axis=1)
        y = df['salary']
        
        # Identificar colunas numéricas e categóricas
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Criar preprocessador
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(with_mean=False), numeric_features),  # with_mean=False para evitar erro com matrizes esparsas
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ])
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logging.info(f"✅ Dados preparados: {X_train.shape} treino, {X_test.shape} teste")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Treinar múltiplos modelos"""
        logging.info("🤖 Treinando modelos de ML...")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
        import joblib
        
        # Definir modelos
        models_config = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Treinar cada modelo
        for name, model in models_config.items():
            logging.info(f"  🔄 Treinando {name}...")
            
            # Criar pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
            
            # Treinar
            pipeline.fit(X_train, y_train)
            
            # Predizer
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            # Armazenar resultados
            self.models[name] = pipeline
            self.results[name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            logging.info(f"  ✅ {name}: Accuracy={accuracy:.4f}, ROC-AUC={roc_auc:.4f}")
        
        # Salvar modelos
        self._save_models()
        
        return self.models, self.results
    
    def _save_models(self):
        """Salvar modelos treinados"""
        import joblib
        
        models_dir = Path("data/processed")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Salvar preprocessador
        joblib.dump(self.preprocessor, models_dir / "preprocessor.joblib")
        
        # Salvar cada modelo
        for name, model in self.models.items():
            filename = f"model_{name.lower().replace(' ', '_')}.joblib"
            joblib.dump(model, models_dir / filename)
        
        logging.info("💾 Modelos salvos em data/processed/")

# =============================================================================
# 4. ANÁLISE DE CLUSTERING
# =============================================================================

class SalaryClusteringAnalysis:
    """Análise de clustering para segmentação de perfis salariais"""
    
    def __init__(self):
        self.kmeans_model = None
        self.dbscan_model = None
        self.pca = None
        self.scaler = None
        
    def perform_kmeans_analysis(self, X, max_clusters=8):
        """Análise K-Means com método do cotovelo"""
        logging.info("🔍 Iniciando análise K-Means...")
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        
        # Tratar matriz esparsa
        if hasattr(X, 'sparse') and X.sparse:
            X_scaled = X.toarray()
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_scaled)
        else:
            try:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            except ValueError as e:
                if "sparse" in str(e).lower():
                    self.scaler = StandardScaler(with_mean=False)
                    X_scaled = self.scaler.fit_transform(X)
                else:
                    raise e
        
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            
            logging.info(f"  📊 K={k}: Inércia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
        
        # Gráfico do cotovelo e silhouette
        self._plot_clustering_analysis(range(2, max_clusters + 1), inertias, silhouette_scores)
        
        # Escolher melhor k
        best_k = range(2, max_clusters + 1)[np.argmax(silhouette_scores)]
        logging.info(f"✅ Melhor número de clusters: {best_k} (Silhouette: {max(silhouette_scores):.3f})")
        
        # Treinar modelo final
        self.kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        clusters = self.kmeans_model.fit_predict(X_scaled)
        
        return clusters, best_k
    
    def _plot_clustering_analysis(self, k_range, inertias, silhouette_scores):
        """Plotar análise do cotovelo e silhouette"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Método do cotovelo
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Número de Clusters')
        ax1.set_ylabel('Inércia')
        ax1.set_title('Método do Cotovelo')
        ax1.grid(True, alpha=0.3)
        
        # Análise silhouette
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Número de Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Análise Silhouette')
        ax2.grid(True, alpha=0.3)
        
        # Marcar melhor k
        best_k_idx = np.argmax(silhouette_scores)
        best_k = k_range[best_k_idx]
        ax2.axvline(best_k, color='green', linestyle='--', alpha=0.7, label=f'Melhor K={best_k}')
        ax2.legend()
        
        plt.tight_layout()
        
        # Salvar
        images_dir = Path("output/images")
        images_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(images_dir / "clustering_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("📈 Gráficos de análise de clustering salvos")
    
    def visualize_clusters_pca(self, X, clusters, target=None):
        """Visualizar clusters com PCA 2D"""
        logging.info("🎨 Gerando visualizações PCA dos clusters...")
        
        from sklearn.decomposition import PCA
        
        # Aplicar PCA
        self.pca = PCA(n_components=2, random_state=42)
        if hasattr(X, 'toarray'):
            X_pca = self.pca.fit_transform(X.toarray())
        else:
            X_pca = self.pca.fit_transform(X)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Clusters
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                                  cmap='viridis', alpha=0.7, s=50)
        axes[0].set_title('Clusters K-Means')
        axes[0].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variância)')
        axes[0].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variância)')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Target real (se disponível)
        if target is not None:
            target_encoded = (target == '>50K').astype(int) if hasattr(target, 'str') else target
            scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=target_encoded, 
                                      cmap='RdYlBu', alpha=0.7, s=50)
            axes[1].set_title('Classes Reais (Salário)')
            axes[1].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variância)')
            axes[1].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variância)')
            plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        
        images_dir = Path("output/images")
        plt.savefig(images_dir / "clusters_pca_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("📈 Visualizações PCA dos clusters salvas")

# =============================================================================
# 5. REGRAS DE ASSOCIAÇÃO
# =============================================================================

class AssociationRulesAnalysis:
    """Análise de regras de associação para padrões salariais"""
    
    def __init__(self):
        self.rules = None
        self.frequent_itemsets = None
    
    def prepare_transaction_data(self, df):
        """Preparar dados para análise de associação"""
        logging.info("🔧 Preparando dados para análise de associação...")
        
        # Verificar se mlxtend está disponível
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            from mlxtend.preprocessing import TransactionEncoder
        except ImportError:
            logging.warning("⚠️ mlxtend não disponível. Execute: pip install mlxtend")
            return []
        
        # Discretizar variáveis numéricas
        df_discrete = df.copy()
        
        # Idade em faixas
        if 'age' in df.columns:
            df_discrete['age_group'] = pd.cut(df['age'], 
                                            bins=[0, 30, 40, 50, 100], 
                                            labels=['jovem', 'adulto', 'maduro', 'senior'])
        
        # Horas em faixas
        if 'hours-per-week' in df.columns:
            df_discrete['hours_group'] = pd.cut(df['hours-per-week'], 
                                              bins=[0, 35, 45, 100], 
                                              labels=['part_time', 'full_time', 'overtime'])
        
        # Educação em faixas
        if 'education-num' in df.columns:
            df_discrete['education_level'] = pd.cut(df['education-num'], 
                                                  bins=[0, 9, 12, 16, 20], 
                                                  labels=['basico', 'medio', 'superior', 'pos_grad'])
        
        # Criar transações
        transactions = []
        for _, row in df_discrete.iterrows():
            transaction = []
            
            # Adicionar características categóricas disponíveis
            categorical_features = [
                ('workclass', 'workclass'),
                ('education', 'education'), 
                ('marital-status', 'marital'),
                ('occupation', 'occupation'),
                ('sex', 'sex'),
                ('age_group', 'age'),
                ('hours_group', 'hours'),
                ('education_level', 'edu_level'),
                ('salary', 'salary')
            ]
            
            for col, prefix in categorical_features:
                if col in row and pd.notna(row[col]):
                    transaction.append(f"{prefix}_{row[col]}")
            
            # Adicionar apenas se temos items suficientes
            if len(transaction) >= 3:
                transactions.append(transaction)
        
        logging.info(f"✅ {len(transactions)} transações preparadas")
        return transactions
    
    def find_association_rules(self, transactions, min_support=0.03, min_confidence=0.5):
        """Encontrar regras de associação"""
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            from mlxtend.preprocessing import TransactionEncoder
        except ImportError:
            logging.error("❌ mlxtend necessário para regras de associação")
            return pd.DataFrame()
        
        if not transactions:
            return pd.DataFrame()
        
        logging.info("🔍 Buscando regras de associação...")
        
        try:
            # Codificar transações
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            
            logging.info(f"  📊 Dataset codificado: {df_encoded.shape}")
            
            # Encontrar itens frequentes
            self.frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            
            if self.frequent_itemsets.empty:
                logging.warning(f"⚠️ Nenhum itemset frequente encontrado com suporte >= {min_support}")
                return pd.DataFrame()
            
            logging.info(f"  📊 {len(self.frequent_itemsets)} itemsets frequentes encontrados")
            
            # Gerar regras
            self.rules = association_rules(self.frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            if self.rules.empty:
                logging.warning(f"⚠️ Nenhuma regra encontrada com confiança >= {min_confidence}")
                return pd.DataFrame()
            
            # Filtrar regras relacionadas com salário
            salary_rules = self.rules[
                self.rules['consequents'].astype(str).str.contains('salary_') |
                self.rules['antecedents'].astype(str).str.contains('salary_')
            ]
            
            logging.info(f"✅ {len(self.rules)} regras totais, {len(salary_rules)} relacionadas a salário")
            
            # Salvar análise
            self._save_rules_analysis(salary_rules)
            
            return salary_rules.sort_values('lift', ascending=False)
            
        except Exception as e:
            logging.error(f"❌ Erro na análise de associação: {e}")
            return pd.DataFrame()
    
    def _save_rules_analysis(self, rules):
        """Salvar análise das regras"""
        output_dir = Path("output/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not rules.empty:
            # Salvar CSV
            rules.to_csv(output_dir / "association_rules_salary.csv", index=False)
            
            # Criar relatório
            report = []
            report.append("# RELATÓRIO DE REGRAS DE ASSOCIAÇÃO\n\n")
            report.append(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            report.append(f"**Total de regras:** {len(rules)}\n\n")
            
            # Top 5 regras
            top_rules = rules.head(5)
            report.append("## TOP 5 REGRAS:\n\n")
            
            for idx, rule in top_rules.iterrows():
                antecedents = ", ".join(list(rule['antecedents']))
                consequents = ", ".join(list(rule['consequents']))
                
                report.append(f"**Regra {idx + 1}:**\n")
                report.append(f"- SE: {antecedents}\n")
                report.append(f"- ENTÃO: {consequents}\n")
                report.append(f"- Confiança: {rule['confidence']:.3f}\n")
                report.append(f"- Lift: {rule['lift']:.3f}\n\n")
            
            # Salvar relatório
            with open(output_dir / "association_rules_report.md", 'w', encoding='utf-8') as f:
                f.writelines(report)
        
        logging.info("📊 Análise de regras de associação salva")

# =============================================================================
# 6. MÉTRICAS AVANÇADAS
# =============================================================================

class AdvancedMetrics:
    """Métricas avançadas para avaliação rigorosa"""
    
    def __init__(self):
        self.metrics_summary = {}
        self.business_kpis = {}
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba, model_name="Model"):
        """Calcular métricas abrangentes"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, precision_recall_curve, auc, roc_curve, confusion_matrix
        )
        
        # Converter para binário se necessário
        if hasattr(y_true, 'iloc') and isinstance(y_true.iloc[0], str):
            y_true_binary = (y_true == '>50K').astype(int)
        else:
            y_true_binary = y_true
            
        if isinstance(y_pred[0], str):
            y_pred_binary = (y_pred == '>50K').astype(int)
        else:
            y_pred_binary = y_pred
        
        # Calcular métricas
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary),
            'recall': recall_score(y_true_binary, y_pred_binary),
            'f1_score': f1_score(y_true_binary, y_pred_binary),
            'roc_auc': roc_auc_score(y_true_binary, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
        }
        
        # Curva Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
        metrics['pr_auc'] = auc(recall, precision)
        
        # Especificidade
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        self.metrics_summary[model_name] = metrics
        
        # Gerar visualizações
        self._plot_roc_pr_curves(y_true_binary, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba, model_name)
        
        logging.info(f"✅ Métricas avançadas calculadas para {model_name}")
        return metrics
    
    def generate_business_kpis(self, df, predictions, model_name="Model"):
        """Gerar KPIs orientados ao negócio"""
        y_true = df['salary']
        y_pred = predictions
        
        kpis = {
            'total_samples': len(df),
            'high_salary_rate': (y_true == '>50K').mean(),
            'prediction_accuracy': (y_pred == y_true).mean(),
            'false_positive_rate': ((y_pred == '>50K') & (y_true == '<=50K')).mean(),
            'false_negative_rate': ((y_pred == '<=50K') & (y_true == '>50K')).mean(),
        }
        
        # KPIs por segmento demográfico
        if 'sex' in df.columns:
            for sex in df['sex'].unique():
                if pd.notna(sex):
                    mask = df['sex'] == sex
                    if mask.sum() > 0:
                        kpis[f'accuracy_{sex.lower()}'] = (y_pred[mask] == y_true.loc[mask]).mean()
        
        self.business_kpis[model_name] = kpis
        self._save_kpi_report(kpis, model_name)
        
        logging.info(f"✅ KPIs de negócio calculados para {model_name}")
        return kpis
    
    def _plot_roc_pr_curves(self, y_true, y_prob, model_name):
        """Plotar curvas ROC e Precision-Recall"""
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('Taxa de Falsos Positivos')
        ax1.set_ylabel('Taxa de Verdadeiros Positivos')
        ax1.set_title(f'Curva ROC - {model_name}')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Curva Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        ax2.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precisão')
        ax2.set_title(f'Curva Precision-Recall - {model_name}')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar
        output_dir = Path("output/images")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"roc_pr_curves_{model_name.lower().replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_kpi_report(self, kpis, model_name):
        """Salvar relatório de KPIs"""
        output_dir = Path("output/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append(f"# RELATÓRIO DE KPIs - {model_name}\n\n")
        report.append(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # KPIs Gerais
        report.append("## 📊 KPIs Gerais\n")
        for key, value in kpis.items():
            if isinstance(value, float):
                report.append(f"- **{key.replace('_', ' ').title()}:** {value:.3f}\n")
            else:
                report.append(f"- **{key.replace('_', ' ').title()}:** {value:,}\n")
        
        # Salvar relatório
        report_file = output_dir / f"kpi_report_{model_name.lower().replace(' ', '_')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.writelines(report)
    
    def generate_comparison_report(self):
        """Gerar relatório comparativo entre modelos"""
        if not self.metrics_summary:
            logging.warning("⚠️ Nenhuma métrica disponível para comparação")
            return
        
        # Criar DataFrame com métricas
        df_metrics = pd.DataFrame(self.metrics_summary).T
        
        # Salvar CSV
        output_dir = Path("output/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        df_metrics.to_csv(output_dir / "model_comparison.csv")
        
        # Gerar gráfico comparativo
        self._plot_model_comparison(df_metrics)
        
        # Gerar relatório em markdown
        self._save_comparison_report(df_metrics)
        
        logging.info("✅ Relatório comparativo gerado")
    
    def _plot_model_comparison(self, df_metrics):
        """Plotar comparação entre modelos"""
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(df_metrics))
        width = 0.15
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in df_metrics.columns:
                ax.bar(x + i * width, df_metrics[metric], width, label=metric.title())
        
        ax.set_xlabel('Modelos')
        ax.set_ylabel('Score')
        ax.set_title('Comparação de Performance entre Modelos')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(df_metrics.index, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = Path("output/images")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_comparison_report(self, df_metrics):
        """Salvar relatório comparativo em markdown"""
        output_dir = Path("output/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append("# RELATÓRIO COMPARATIVO DE MODELOS\n\n")
        report.append(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # Tabela de métricas
        report.append("## 📊 Comparação de Métricas\n\n")
        report.append("| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n")
        report.append("|--------|----------|-----------|--------|----------|----------|\n")
        
        for model_name, row in df_metrics.iterrows():
            report.append(f"| {model_name} |")
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                if metric in row:
                    report.append(f" {row[metric]:.4f} |")
                else:
                    report.append(" N/A |")
            report.append("\n")
        
        # Melhor modelo por métrica
        report.append("\n## 🏆 Melhor Modelo por Métrica\n\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            if metric in df_metrics.columns:
                best_model = df_metrics[metric].idxmax()
                best_score = df_metrics[metric].max()
                report.append(f"- **{metric.title()}**: {best_model} ({best_score:.4f})\n")
        
        # Salvar relatório
        report_file = output_dir / "model_comparison_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        logging.info(f"📊 Relatório comparativo salvo: {report_file}")

# =============================================================================
# 7. PIPELINE PRINCIPAL
# =============================================================================

def main():
    """Pipeline principal completo - Versão Académica Final"""
    logging.info("🚀 Iniciando Sistema Completo de Análise Salarial - Versão Académica Final")
    logging.info("="*80)
    
    try:
        # 1. Processamento de Dados
        logging.info("📊 FASE 1: PROCESSAMENTO DE DADOS")
        logging.info("="*50)
        
        processor = DataProcessor()
        df = processor.load_data()
        df = processor.clean_data()
        df = processor.create_features()
        
        # 2. Análise Exploratória
        logging.info("\n📈 FASE 2: ANÁLISE EXPLORATÓRIA")
        logging.info("="*50)
        
        explorer = ExploratoryAnalysis(df)
        explorer.generate_basic_statistics()
        explorer.create_visualizations()
        
        # 3. Modelagem de Machine Learning
        logging.info("\n🤖 FASE 3: MODELAGEM DE MACHINE LEARNING")
        logging.info("="*50)
        
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(df)
        models, results = trainer.train_models(X_train, X_test, y_train, y_test)
        
        # 4. Análise de Clustering
        logging.info("\n🎯 FASE 4: ANÁLISE DE CLUSTERING")
        logging.info("="*50)
        
        clustering = SalaryClusteringAnalysis()
        
        # Usar preprocessador salvo
        import joblib
        preprocessor_path = Path("data/processed/preprocessor.joblib")
        if preprocessor_path.exists():
            try:
                preprocessor = joblib.load(preprocessor_path)
                X_features = df.drop('salary', axis=1)
                X_processed = preprocessor.transform(X_features)
                
                clusters, best_k = clustering.perform_kmeans_analysis(X_processed)
                clustering.visualize_clusters_pca(X_processed, clusters, df['salary'])
                
                logging.info(f"✅ Clustering concluído: {best_k} clusters identificados")
            except Exception as e:
                logging.error(f"❌ Erro no clustering: {e}")
        else:
            logging.warning("⚠️ Preprocessador não encontrado")
        
        # 5. Regras de Associação
        logging.info("\n📋 FASE 5: REGRAS DE ASSOCIAÇÃO")
        logging.info("="*50)
        
        association = AssociationRulesAnalysis()
        transactions = association.prepare_transaction_data(df)
        
        if transactions:
            rules = association.find_association_rules(transactions, min_support=0.03, min_confidence=0.5)
            logging.info(f"✅ {len(rules)} regras de associação encontradas")
        else:
            logging.warning("⚠️ Nenhuma transação válida para análise")
            rules = pd.DataFrame()
        
        # 6. Métricas Avançadas
        logging.info("\n📊 FASE 6: MÉTRICAS AVANÇADAS")
        logging.info("="*50)
        
        advanced_metrics = AdvancedMetrics()
        
        # Calcular métricas para cada modelo
        for model_name, result in results.items():
            if all(key in result for key in ['y_test', 'y_pred', 'y_pred_proba']):
                try:
                    metrics = advanced_metrics.calculate_comprehensive_metrics(
                        result['y_test'], 
                        result['y_pred'], 
                        result['y_pred_proba'],
                        model_name
                    )
                    
                    # Gerar KPIs de negócio
                    kpis = advanced_metrics.generate_business_kpis(df, result['y_pred'], model_name)
                    
                    logging.info(f"✅ Métricas avançadas calculadas para {model_name}")
                except Exception as e:
                    logging.error(f"❌ Erro ao calcular métricas para {model_name}: {e}")
        
        # Gerar relatório comparativo
        try:
            advanced_metrics.generate_comparison_report()
            logging.info("✅ Relatório comparativo gerado")
        except Exception as e:
            logging.error(f"❌ Erro ao gerar relatório comparativo: {e}")
        
        # 7. Relatório Final
        logging.info("\n📋 RELATÓRIO FINAL COMPLETO")
        logging.info("="*80)
        
        logging.info(f"📊 Dataset final: {len(df)} registros, {len(df.columns)} colunas")
        logging.info("🤖 Modelos treinados:")
        for name, result in results.items():
            logging.info(f"  • {name}: Accuracy={result['accuracy']:.4f}")
        
        if 'best_k' in locals():
            logging.info(f"🎯 Clustering: {best_k} clusters identificados")
        
        logging.info(f"📋 Regras de associação: {len(rules) if not rules.empty else 0}")
        
        logging.info("\n📂 Estrutura de saídas geradas:")
        logging.info("  📈 Visualizações: output/images/")
        logging.info("  📊 Dados processados: data/processed/")
        logging.info("  🤖 Modelos (.joblib): data/processed/")
        logging.info("  📋 Análises avançadas: output/analysis/")
        logging.info("  📝 Relatórios: output/")
        
        # Listar arquivos gerados
        _list_generated_files()
        
        logging.info("\n🎉 PIPELINE ACADÉMICO COMPLETO CONCLUÍDO COM SUCESSO!")
        logging.info("🎓 Projeto pronto para avaliação académica de mestrado!")
        logging.info("\n💡 Próximos passos:")
        logging.info("  1. Execute: streamlit run dashboard_app.py")
        logging.info("  2. Revise os relatórios em output/analysis/")
        logging.info("  3. Analise as visualizações em output/images/")
        
    except Exception as e:
        logging.error(f"❌ Erro crítico durante execução: {e}")
        raise

def _list_generated_files():
    """Listar todos os arquivos gerados"""
    
    # Arquivos .joblib
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        joblib_files = list(processed_dir.glob("*.joblib"))
        if joblib_files:
            logging.info("\n📁 Modelos e preprocessadores (.joblib):")
            for file in joblib_files:
                logging.info(f"  • {file.name}")
    
    # Visualizações
    images_dir = Path("output/images")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.png"))
        if image_files:
            logging.info(f"\n🎨 Visualizações geradas:")
            for img in sorted(image_files):
                logging.info(f"  • {img.name}")
    
    # Análises
    analysis_dir = Path("output/analysis")
    if analysis_dir.exists():
        analysis_files = list(analysis_dir.glob("*"))
        if analysis_files:
            logging.info(f"\n🔍 Análises e relatórios:")
            for file in sorted(analysis_files):
                logging.info(f"  • {file.name}")

if __name__ == "__main__":
    main()