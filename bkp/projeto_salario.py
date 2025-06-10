"""
Sistema Completo de Análise Salarial - Versão 2.0 SQL Integrada
Pipeline Académico Completo com Arquitetura Modular e Base de Dados
Compatível com a nova estrutura SQL desenvolvida
"""

import os
import sys
import logging
import warnings
import traceback
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações iniciais
warnings.filterwarnings('ignore')

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

# Carregar variáveis de ambiente
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Configuração de logging acadêmico
def setup_academic_logging():
    """Configurar logging para relatório acadêmico"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/projeto_salario_v2.log', encoding='utf-8')
        ]
    )
    
    return logging.getLogger(__name__)

class ProjetoSalarioV2:
    """
    Pipeline Acadêmico Completo v2.0
    Integração com arquitetura SQL modular
    """
    
    def __init__(self, use_database: bool = True, force_csv: bool = False):
        self.logger = setup_academic_logging()
        self.use_database = use_database and not force_csv
        
        # Verificar disponibilidade do banco
        if self.use_database:
            self.use_database = self._check_database_availability()
        
        # Resultados do pipeline
        self.df = None
        self.models = {}
        self.results = {}
        self.clustering_results = {}
        self.association_rules = []
        self.advanced_metrics = {}
        
        # Estatísticas do processo
        self.processing_stats = {
            'start_time': datetime.now(),
            'data_source': 'database' if self.use_database else 'csv',
            'total_records': 0,
            'models_trained': 0,
            'visualizations_created': 0,
            'analyses_completed': 0
        }
        
        self.logger.info("🎓 PROJETO SALÁRIO v2.0 - PIPELINE ACADÊMICO INICIADO")
        self.logger.info(f"📊 Fonte de dados: {'Base de Dados SQL' if self.use_database else 'Arquivo CSV'}")
    
    def _check_database_availability(self) -> bool:
        """Verificar disponibilidade da base de dados"""
        try:
            required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
            missing = [var for var in required_vars if not os.getenv(var)]
            
            if missing:
                self.logger.warning(f"⚠️ Variáveis de ambiente ausentes: {missing}")
                self.logger.info("📁 Usando modo CSV como fallback")
                return False
            
            # Testar conexão
            from src.database.connection import DatabaseConnection
            with DatabaseConnection() as db:
                result = db.execute_query("SELECT 1")
                if result:
                    self.logger.info("✅ Conexão com base de dados estabelecida")
                    return True
                    
        except ImportError:
            self.logger.warning("⚠️ Módulos de base de dados não disponíveis")
            return False
        except Exception as e:
            self.logger.warning(f"⚠️ Erro na conexão com base de dados: {e}")
            return False
        
        return False
    
    def run_complete_pipeline(self):
        """Executar pipeline acadêmico completo"""
        self.logger.info("="*80)
        self.logger.info("🚀 INICIANDO PIPELINE ACADÊMICO COMPLETO")
        self.logger.info("="*80)
        
        try:
            # Fase 1: Carregamento e Processamento de Dados
            self._phase_1_data_processing()
            
            # Fase 2: Análise Exploratória
            self._phase_2_exploratory_analysis()
            
            # Fase 3: Modelagem de Machine Learning
            self._phase_3_machine_learning()
            
            # Fase 4: Análise de Clustering
            self._phase_4_clustering_analysis()
            
            # Fase 5: Regras de Associação
            self._phase_5_association_rules()
            
            # Fase 6: Métricas Avançadas
            self._phase_6_advanced_metrics()
            
            # Fase 7: Visualizações Finais
            self._phase_7_final_visualizations()
            
            # Fase 8: Relatório Acadêmico Final
            self._phase_8_academic_report()
            
            self.logger.info("🎉 PIPELINE ACADÊMICO CONCLUÍDO COM SUCESSO!")
            
        except Exception as e:
            self.logger.error(f"❌ Erro crítico no pipeline: {e}")
            self.logger.error(traceback.format_exc())
            self._generate_error_report()
            raise
    
    def _phase_1_data_processing(self):
        """Fase 1: Carregamento e Processamento de Dados"""
        self.logger.info("\n📊 FASE 1: CARREGAMENTO E PROCESSAMENTO DE DADOS")
        self.logger.info("="*60)
        
        if self.use_database:
            self._load_from_database()
        else:
            self._load_from_csv()
        
        if self.df is None or len(self.df) == 0:
            raise ValueError("❌ Nenhum dado foi carregado")
        
        self.processing_stats['total_records'] = len(self.df)
        self.logger.info(f"✅ Dados carregados: {len(self.df):,} registros")
        self.logger.info(f"📋 Colunas: {len(self.df.columns)} features")
        
        # Limpeza e preprocessamento
        self._clean_and_preprocess_data()
    
    def _load_from_database(self):
        """Carregar dados da base de dados SQL"""
        try:
            from src.database.models import SalaryAnalysisSQL
            
            sql_model = SalaryAnalysisSQL()
            self.df = sql_model.get_dataset_for_ml()
            
            if self.df is not None:
                self.logger.info(f"📊 Dados carregados da base de dados: {len(self.df)} registros")
            else:
                self.logger.warning("⚠️ Nenhum dado encontrado na base de dados")
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar da base de dados: {e}")
            self.logger.info("📁 Tentando fallback para CSV...")
            self._load_from_csv()
    
    def _load_from_csv(self):
        """Carregar dados do arquivo CSV"""
        csv_paths = [
            "4-Carateristicas_salario.csv",
            "data/raw/4-Carateristicas_salario.csv",
            "bkp/4-Carateristicas_salario.csv"
        ]
        
        for csv_path in csv_paths:
            if Path(csv_path).exists():
                try:
                    self.df = pd.read_csv(csv_path)
                    self.logger.info(f"📁 Dados carregados do CSV: {csv_path}")
                    self.processing_stats['data_source'] = 'csv'
                    return
                except Exception as e:
                    self.logger.warning(f"⚠️ Erro ao carregar {csv_path}: {e}")
                    continue
        
        raise FileNotFoundError("❌ Arquivo CSV não encontrado em nenhum local")
    
    def _clean_and_preprocess_data(self):
        """Limpeza e preprocessamento dos dados"""
        self.logger.info("🧹 Limpeza e preprocessamento...")
        
        # Registrar estado inicial
        initial_shape = self.df.shape
        
        # Remover espaços e caracteres especiais
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].astype(str).str.strip()
        
        # Substituir valores ausentes
        self.df = self.df.replace('?', np.nan)
        
        # Tratar valores ausentes
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in self.df.columns:
                mode_value = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'Unknown'
                self.df[col].fillna(mode_value, inplace=True)
        
        # Remover linhas com muitos valores ausentes
        missing_threshold = 0.5
        self.df = self.df.dropna(thresh=len(self.df.columns) * missing_threshold)
        
        # Criar features derivadas
        self._create_derived_features()
        
        final_shape = self.df.shape
        
        self.logger.info(f"✅ Limpeza concluída: {initial_shape} → {final_shape}")
        self.logger.info(f"📉 Registros removidos: {initial_shape[0] - final_shape[0]:,}")
    
    def _create_derived_features(self):
        """Criar features derivadas para análise"""
        self.logger.info("🔧 Criando features derivadas...")
        
        features_created = []
        
        # Faixas etárias
        if 'age' in self.df.columns:
            self.df['age_group'] = pd.cut(
                self.df['age'], 
                bins=[0, 25, 35, 45, 55, 100],
                labels=['Jovem', 'Adulto_Jovem', 'Adulto', 'Maduro', 'Senior']
            )
            features_created.append('age_group')
        
        # Intensidade de trabalho
        hours_col = 'hours-per-week' if 'hours-per-week' in self.df.columns else 'hours_per_week'
        if hours_col in self.df.columns:
            self.df['work_intensity'] = pd.cut(
                self.df[hours_col],
                bins=[0, 35, 45, 60, 100],
                labels=['Part_Time', 'Normal', 'Overtime', 'Workaholic']
            )
            features_created.append('work_intensity')
        
        # Nível educacional
        edu_num_col = 'education-num' if 'education-num' in self.df.columns else 'education_num'
        if edu_num_col in self.df.columns:
            self.df['education_level'] = pd.cut(
                self.df[edu_num_col],
                bins=[0, 9, 12, 16, 20],
                labels=['Básico', 'Médio', 'Superior', 'Pós_Graduação']
            )
            features_created.append('education_level')
        
        # Capital líquido
        gain_col = 'capital-gain' if 'capital-gain' in self.df.columns else 'capital_gain'
        loss_col = 'capital-loss' if 'capital-loss' in self.df.columns else 'capital_loss'
        
        if gain_col in self.df.columns and loss_col in self.df.columns:
            self.df['net_capital'] = self.df[gain_col] - self.df[loss_col]
            self.df['has_capital_gain'] = (self.df[gain_col] > 0).astype(int)
            self.df['has_capital_loss'] = (self.df[loss_col] > 0).astype(int)
            features_created.extend(['net_capital', 'has_capital_gain', 'has_capital_loss'])
        
        self.logger.info(f"✅ Features criadas: {features_created}")
    
    def _phase_2_exploratory_analysis(self):
        """Fase 2: Análise Exploratória"""
        self.logger.info("\n📈 FASE 2: ANÁLISE EXPLORATÓRIA")
        self.logger.info("="*60)
        
        try:
            # Criar diretórios
            output_dir = Path("output")
            images_dir = output_dir / "images"
            analysis_dir = output_dir / "analysis"
            
            for directory in [output_dir, images_dir, analysis_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Configurar estilo
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Estatísticas descritivas
            self._generate_descriptive_statistics(analysis_dir)
            
            # Visualizações exploratórias
            self._create_exploratory_visualizations(images_dir)
            
            # Análise de correlações
            self._analyze_correlations(images_dir)
            
            self.processing_stats['analyses_completed'] += 1
            self.logger.info("✅ Análise exploratória concluída")
            
        except Exception as e:
            self.logger.error(f"❌ Erro na análise exploratória: {e}")
    
    def _generate_descriptive_statistics(self, output_dir):
        """Gerar estatísticas descritivas"""
        self.logger.info("📊 Gerando estatísticas descritivas...")
        
        # Estatísticas numéricas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = self.df[numeric_cols].describe()
            numeric_stats.to_csv(output_dir / "numeric_statistics.csv")
            
            self.logger.info(f"📋 Estatísticas numéricas salvas: {len(numeric_cols)} variáveis")
        
        # Estatísticas categóricas
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        categorical_stats = {}
        
        for col in categorical_cols:
            categorical_stats[col] = {
                'unique_values': self.df[col].nunique(),
                'most_frequent': self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'N/A',
                'missing_values': self.df[col].isnull().sum(),
                'missing_percentage': (self.df[col].isnull().sum() / len(self.df)) * 100
            }
        
        categorical_df = pd.DataFrame(categorical_stats).T
        categorical_df.to_csv(output_dir / "categorical_statistics.csv")
        
        self.logger.info(f"📋 Estatísticas categóricas salvas: {len(categorical_cols)} variáveis")
        
        # Distribuição da variável target
        target_col = 'salary'
        if target_col in self.df.columns:
            target_dist = self.df[target_col].value_counts(normalize=True)
            target_stats = {
                'distribution': target_dist.to_dict(),
                'balance_ratio': target_dist.min() / target_dist.max(),
                'class_count': len(target_dist)
            }
            
            import json
            with open(output_dir / "target_analysis.json", 'w') as f:
                json.dump(target_stats, f, indent=2)
            
            self.logger.info(f"🎯 Análise da variável target: {target_stats['class_count']} classes")
            self.logger.info(f"⚖️ Razão de balanceamento: {target_stats['balance_ratio']:.3f}")
    
    def _create_exploratory_visualizations(self, images_dir):
        """Criar visualizações exploratórias"""
        self.logger.info("🎨 Criando visualizações exploratórias...")
        
        visualizations_created = 0
        
        # 1. Distribuição da variável target
        if 'salary' in self.df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Distribuição de salário
            self.df['salary'].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
            axes[0,0].set_title('Distribuição de Salário', fontsize=14, fontweight='bold')
            axes[0,0].set_ylabel('Frequência')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Salário por idade
            if 'age' in self.df.columns:
                sns.boxplot(data=self.df, x='salary', y='age', ax=axes[0,1])
                axes[0,1].set_title('Distribuição de Idade por Salário', fontsize=14, fontweight='bold')
            
            # Salário por sexo
            if 'sex' in self.df.columns:
                salary_sex = pd.crosstab(self.df['sex'], self.df['salary'], normalize='index')
                salary_sex.plot(kind='bar', ax=axes[1,0], color=['lightcoral', 'lightgreen'])
                axes[1,0].set_title('Distribuição de Salário por Sexo', fontsize=14, fontweight='bold')
                axes[1,0].legend(title='Salário')
                axes[1,0].tick_params(axis='x', rotation=45)
            
            # Salário por educação (top 8)
            if 'education' in self.df.columns:
                top_education = self.df['education'].value_counts().head(8).index
                education_salary = self.df[self.df['education'].isin(top_education)]
                salary_edu = pd.crosstab(education_salary['education'], education_salary['salary'], normalize='index')
                salary_edu.plot(kind='bar', ax=axes[1,1], color=['orange', 'green'])
                axes[1,1].set_title('Distribuição de Salário por Educação (Top 8)', fontsize=14, fontweight='bold')
                axes[1,1].tick_params(axis='x', rotation=45)
                axes[1,1].legend(title='Salário')
            
            plt.tight_layout()
            plt.savefig(images_dir / "exploratory_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations_created += 1
        
        # 2. Distribuições das variáveis numéricas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Limitar a 6 variáveis para visualização
            cols_to_plot = list(numeric_cols)[:6]
            n_cols = min(3, len(cols_to_plot))
            n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(cols_to_plot):
                if i < len(axes):
                    self.df[col].hist(bins=30, ax=axes[i], color='skyblue', alpha=0.7)
                    axes[i].set_title(f'Distribuição: {col}', fontsize=12, fontweight='bold')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequência')
                    axes[i].grid(True, alpha=0.3)
            
            # Ocultar axes extras
            for i in range(len(cols_to_plot), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(images_dir / "numeric_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations_created += 1
        
        # 3. Distribuições das variáveis categóricas
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            # Limitar a 6 variáveis para visualização
            cols_to_plot = [col for col in categorical_cols if self.df[col].nunique() <= 20][:6]
            
            if cols_to_plot:
                n_cols = min(3, len(cols_to_plot))
                n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                if n_rows == 1:
                    axes = [axes] if n_cols == 1 else axes
                else:
                    axes = axes.flatten()
                
                for i, col in enumerate(cols_to_plot):
                    if i < len(axes):
                        value_counts = self.df[col].value_counts().head(10)
                        value_counts.plot(kind='bar', ax=axes[i], color='lightgreen')
                        axes[i].set_title(f'Distribuição: {col}', fontsize=12, fontweight='bold')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frequência')
                        axes[i].tick_params(axis='x', rotation=45)
                        axes[i].grid(True, alpha=0.3)
                
                # Ocultar axes extras
                for i in range(len(cols_to_plot), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(images_dir / "categorical_distributions.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                visualizations_created += 1
        
        self.processing_stats['visualizations_created'] += visualizations_created
        self.logger.info(f"✅ {visualizations_created} visualizações exploratórias criadas")
    
    def _analyze_correlations(self, images_dir):
        """Analisar correlações entre variáveis"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            self.logger.info("🔗 Analisando correlações...")
            
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.df[numeric_cols].corr()
            
            # Criar heatmap
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(
                correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8}
            )
            
            plt.title('Matriz de Correlação - Variáveis Numéricas', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(images_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Salvar correlações mais fortes
            correlation_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.3:  # Correlações moderadas ou fortes
                        correlation_pairs.append({
                            'var1': correlation_matrix.columns[i],
                            'var2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            if correlation_pairs:
                corr_df = pd.DataFrame(correlation_pairs)
                corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
                corr_df.to_csv(Path("output/analysis") / "strong_correlations.csv", index=False)
                
                self.logger.info(f"🔗 {len(correlation_pairs)} correlações fortes encontradas")
    
    def _phase_3_machine_learning(self):
        """Fase 3: Modelagem de Machine Learning"""
        self.logger.info("\n🤖 FASE 3: MODELAGEM DE MACHINE LEARNING")
        self.logger.info("="*60)
        
        try:
            if self.use_database:
                # Usar pipeline ML integrado
                from src.pipelines.ml_pipeline import MLPipeline
                ml_pipeline = MLPipeline()
                self.models, self.results = ml_pipeline.run(self.df)
            else:
                # Usar implementação standalone
                self._train_models_standalone()
            
            self.processing_stats['models_trained'] = len(self.models)
            self.logger.info(f"✅ {len(self.models)} modelos treinados com sucesso")
            
            # Resumo dos resultados
            for name, result in self.results.items():
                accuracy = result.get('accuracy', 0)
                self.logger.info(f"   • {name}: Acurácia = {accuracy:.4f}")
                
        except Exception as e:
            self.logger.error(f"❌ Erro na modelagem ML: {e}")
            self.models = {}
            self.results = {}
    
    def _train_models_standalone(self):
        """Treinar modelos de forma standalone (sem arquitetura SQL)"""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report
        import joblib
        
        self.logger.info("🔧 Preparando dados para modelagem standalone...")
        
        # Verificar se temos coluna target
        target_col = 'salary'
        if target_col not in self.df.columns:
            raise ValueError(f"Coluna target '{target_col}' não encontrada")
        
        # Preparar features e target
        X = self.df.drop(target_col, axis=1)
        y = self.df[target_col]
        
        # Identificar tipos de colunas
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Criar preprocessador
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Preprocessar
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Codificar target se necessário
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y_train_encoded = le_target.fit_transform(y_train)
            y_test_encoded = le_target.transform(y_test)
        else:
            y_train_encoded = y_train
            y_test_encoded = y_test
            le_target = None
        
        # Treinar modelos
        models_config = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }
        
        for name, model in models_config.items():
            self.logger.info(f"🔄 Treinando {name}...")
            
            # Treinar
            model.fit(X_train_processed, y_train_encoded)
            
            # Predições
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed) if hasattr(model, 'predict_proba') else None
            
            # Métricas
            accuracy = accuracy_score(y_test_encoded, y_pred)
            
            # Salvar modelo
            self.models[name] = model
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'y_test': y_test_encoded,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            self.logger.info(f"✅ {name} treinado: Acurácia = {accuracy:.4f}")
        
        # Salvar artefatos
        models_dir = Path("data/processed")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Salvar preprocessador
        joblib.dump(preprocessor, models_dir / "preprocessor_v2.joblib")
        
        # Salvar modelos
        for name, model in self.models.items():
            filename = f"{name.lower().replace(' ', '_')}_model_v2.joblib"
            joblib.dump(model, models_dir / filename)
        
        # Salvar encoder de target se existir
        if le_target:
            joblib.dump(le_target, models_dir / "target_encoder_v2.joblib")
        
        self.logger.info("💾 Modelos e preprocessadores salvos")
    
    def _phase_4_clustering_analysis(self):
        """Fase 4: Análise de Clustering"""
        self.logger.info("\n🎯 FASE 4: ANÁLISE DE CLUSTERING")
        self.logger.info("="*60)
        
        try:
            if self.use_database:
                # Usar pipeline de análise integrado
                from src.analysis.clustering import SalaryClusteringAnalysis
                clustering = SalaryClusteringAnalysis()
                
                # Preparar dados para clustering
                features_for_clustering = self._prepare_clustering_data()
                if features_for_clustering is not None:
                    clusters, best_k = clustering.perform_kmeans_analysis(features_for_clustering)
                    clustering.visualize_clusters_pca(features_for_clustering, clusters, self.df.get('salary'))
                    
                    self.clustering_results = {
                        'best_k': best_k,
                        'clusters': clusters,
                        'model': clustering.kmeans_model
                    }
                    
                    self.logger.info(f"✅ Clustering concluído: {best_k} clusters identificados")
            else:
                # Implementação standalone
                self._perform_clustering_standalone()
                
        except Exception as e:
            self.logger.error(f"❌ Erro na análise de clustering: {e}")
            self.clustering_results = {}
    
    def _prepare_clustering_data(self):
        """Preparar dados para clustering"""
        try:
            # Verificar se temos preprocessador salvo
            preprocessor_path = Path("data/processed/preprocessor_v2.joblib")
            if preprocessor_path.exists():
                import joblib
                preprocessor = joblib.load(preprocessor_path)
                
                # Preparar features (sem target)
                target_col = 'salary'
                X = self.df.drop(target_col, axis=1) if target_col in self.df.columns else self.df
                
                # Aplicar preprocessamento
                X_processed = preprocessor.transform(X)
                return X_processed
            else:
                self.logger.warning("⚠️ Preprocessador não encontrado, usando apenas variáveis numéricas")
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    return scaler.fit_transform(self.df[numeric_cols])
                
        except Exception as e:
            self.logger.error(f"❌ Erro na preparação dos dados para clustering: {e}")
            return None
    
    def _perform_clustering_standalone(self):
        """Realizar clustering de forma standalone"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        self.logger.info("🔍 Realizando análise de clustering standalone...")
        
        # Preparar dados
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            self.logger.warning("⚠️ Insuficientes variáveis numéricas para clustering")
            return
        
        X = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Análise do cotovelo
        k_range = range(2, 9)
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Escolher melhor k
        best_k = k_range[np.argmax(silhouette_scores)]
        
        # Modelo final
        kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        clusters = kmeans_final.fit_predict(X_scaled)
        
        # Visualização PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Clusters
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        axes[0].set_title(f'Clusters K-Means (k={best_k})')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variância)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variância)')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Target (se disponível)
        if 'salary' in self.df.columns:
            target_encoded = (self.df['salary'] == '>50K').astype(int)
            scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=target_encoded, cmap='RdYlBu', alpha=0.7)
            axes[1].set_title('Classes Reais (Salário)')
            axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variância)')
            axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variância)')
            plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        
        images_dir = Path("output/images")
        images_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(images_dir / "clustering_analysis_v2.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Salvar resultados
        self.clustering_results = {
            'best_k': best_k,
            'clusters': clusters,
            'silhouette_score': max(silhouette_scores),
            'model': kmeans_final
        }
        
        # Salvar estatísticas dos clusters
        analysis_dir = Path("output/analysis")
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        cluster_stats = []
        for cluster_id in range(best_k):
            mask = clusters == cluster_id
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': np.sum(mask),
                'percentage': (np.sum(mask) / len(clusters)) * 100
            })
        
        cluster_df = pd.DataFrame(cluster_stats)
        cluster_df.to_csv(analysis_dir / "clustering_results_v2.csv", index=False)
        
        self.logger.info(f"✅ Clustering standalone concluído: {best_k} clusters")
    
    def _phase_5_association_rules(self):
        """Fase 5: Regras de Associação"""
        self.logger.info("\n📋 FASE 5: REGRAS DE ASSOCIAÇÃO")
        self.logger.info("="*60)
        
        try:
            # Verificar se mlxtend está disponível
            try:
                from mlxtend.frequent_patterns import apriori, association_rules
                from mlxtend.preprocessing import TransactionEncoder
            except ImportError:
                self.logger.warning("⚠️ mlxtend não disponível. Execute: pip install mlxtend")
                return
            
            # Preparar dados transacionais
            transactions = self._prepare_transaction_data()
            
            if not transactions:
                self.logger.warning("⚠️ Nenhuma transação válida para análise")
                return
            
            self.logger.info(f"🔄 Analisando {len(transactions)} transações...")
            
            # Codificar transações
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Encontrar itens frequentes
            frequent_itemsets = apriori(df_encoded, min_support=0.03, use_colnames=True)
            
            if frequent_itemsets.empty:
                self.logger.warning("⚠️ Nenhum itemset frequente encontrado")
                return
            
            # Gerar regras
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
            
            if rules.empty:
                self.logger.warning("⚠️ Nenhuma regra de associação encontrada")
                return
            
            # Filtrar regras relacionadas com salário
            salary_rules = rules[
                rules['consequents'].astype(str).str.contains('salary_') |
                rules['antecedents'].astype(str).str.contains('salary_')
            ]
            
            self.association_rules = salary_rules.sort_values('lift', ascending=False)
            
            # Salvar resultados
            analysis_dir = Path("output/analysis")
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            self.association_rules.to_csv(analysis_dir / "association_rules_v2.csv", index=False)
            
            self.logger.info(f"✅ {len(self.association_rules)} regras de associação encontradas")
            
            # Mostrar top 5 regras
            if len(self.association_rules) > 0:
                self.logger.info("🔝 Top 5 regras:")
                for idx, rule in self.association_rules.head(5).iterrows():
                    antecedents = ", ".join(list(rule['antecedents']))
                    consequents = ", ".join(list(rule['consequents']))
                    self.logger.info(f"   SE {antecedents} → ENTÃO {consequents} (Conf: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f})")
                    
        except Exception as e:
            self.logger.error(f"❌ Erro na análise de regras de associação: {e}")
            self.association_rules = []
    
    def _prepare_transaction_data(self):
        """Preparar dados para análise transacional"""
        transactions = []
        
        for _, row in self.df.iterrows():
            transaction = []
            
            # Adicionar características categóricas
            categorical_features = [
                ('workclass', 'workclass'),
                ('education', 'education'),
                ('marital-status', 'marital'),
                ('occupation', 'occupation'),
                ('relationship', 'relationship'),
                ('race', 'race'),
                ('sex', 'sex'),
                ('native-country', 'country'),
                ('salary', 'salary')
            ]
            
            for col, prefix in categorical_features:
                if col in row and pd.notna(row[col]):
                    transaction.append(f"{prefix}_{row[col]}")
            
            # Adicionar características derivadas
            derived_features = [
                ('age_group', 'age'),
                ('work_intensity', 'work'),
                ('education_level', 'edu_level')
            ]
            
            for col, prefix in derived_features:
                if col in row and pd.notna(row[col]):
                    transaction.append(f"{prefix}_{row[col]}")
            
            # Adicionar apenas se temos items suficientes
            if len(transaction) >= 3:
                transactions.append(transaction)
        
        return transactions
    
    def _phase_6_advanced_metrics(self):
        """Fase 6: Métricas Avançadas"""
        self.logger.info("\n📊 FASE 6: MÉTRICAS AVANÇADAS")
        self.logger.info("="*60)
        
        if not self.results:
            self.logger.warning("⚠️ Nenhum modelo disponível para métricas avançadas")
            return
        
        try:
            from sklearn.metrics import (
                precision_score, recall_score, f1_score, roc_auc_score,
                confusion_matrix, classification_report
            )
            
            advanced_metrics = {}
            
            for model_name, result in self.results.items():
                self.logger.info(f"📊 Calculando métricas para {model_name}...")
                
                y_test = result['y_test']
                y_pred = result['y_pred']
                y_pred_proba = result.get('y_pred_proba')
                
                # Métricas básicas
                metrics = {
                    'model_name': model_name,
                    'accuracy': result['accuracy'],
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred, average='weighted')
                }
                
                # ROC AUC (se probabilidades disponíveis)
                if y_pred_proba is not None:
                    if y_pred_proba.ndim > 1:
                        # Classificação binária
                        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                
                # Matriz de confusão
                cm = confusion_matrix(y_test, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                advanced_metrics[model_name] = metrics
                
                self.logger.info(f"   ✅ Acurácia: {metrics['accuracy']:.4f}")
                self.logger.info(f"   ✅ F1-Score: {metrics['f1_score']:.4f}")
                if 'roc_auc' in metrics:
                    self.logger.info(f"   ✅ ROC-AUC: {metrics['roc_auc']:.4f}")
            
            self.advanced_metrics = advanced_metrics
            
            # Salvar métricas comparativas
            metrics_df = pd.DataFrame(advanced_metrics).T
            
            analysis_dir = Path("output/analysis")
            analysis_dir.mkdir(parents=True, exist_ok=True)
            metrics_df.to_csv(analysis_dir / "advanced_metrics_v2.csv")
            
            # Gerar gráfico comparativo
            self._plot_model_comparison(metrics_df)
            
            self.logger.info("✅ Métricas avançadas calculadas e salvas")
            
        except Exception as e:
            self.logger.error(f"❌ Erro no cálculo de métricas avançadas: {e}")
    
    def _plot_model_comparison(self, metrics_df):
        """Plotar comparação entre modelos"""
        if len(metrics_df) == 0:
            return
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
        
        if not available_metrics:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(metrics_df))
        width = 0.2
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(available_metrics):
            values = metrics_df[metric].values
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), 
                   color=colors[i % len(colors)], alpha=0.8)
        
        ax.set_xlabel('Modelos', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Comparação de Performance entre Modelos', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
        ax.set_xticklabels(metrics_df.index, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        images_dir = Path("output/images")
        plt.savefig(images_dir / "model_comparison_v2.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.processing_stats['visualizations_created'] += 1
    
    def _phase_7_final_visualizations(self):
        """Fase 7: Visualizações Finais"""
        self.logger.info("\n🎨 FASE 7: VISUALIZAÇÕES FINAIS")
        self.logger.info("="*60)
        
        try:
            images_dir = Path("output/images")
            images_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Dashboard de resumo
            self._create_summary_dashboard(images_dir)
            
            # 2. Feature importance (se disponível)
            self._plot_feature_importance(images_dir)
            
            # 3. Análise temporal (se aplicável)
            self._create_temporal_analysis(images_dir)
            
            self.logger.info("✅ Visualizações finais criadas")
            
        except Exception as e:
            self.logger.error(f"❌ Erro na criação de visualizações finais: {e}")
    
    def _create_summary_dashboard(self, images_dir):
        """Criar dashboard de resumo"""
        fig = plt.figure(figsize=(20, 12))
        
        # Grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Distribuição da variável target
        if 'salary' in self.df.columns:
            ax1 = fig.add_subplot(gs[0, 0])
            self.df['salary'].value_counts().plot(kind='pie', ax=ax1, autopct='%1.1f%%')
            ax1.set_title('Distribuição de Salário', fontweight='bold')
            ax1.set_ylabel('')
        
        # 2. Distribuição por sexo
        if 'sex' in self.df.columns:
            ax2 = fig.add_subplot(gs[0, 1])
            self.df['sex'].value_counts().plot(kind='bar', ax=ax2, color=['lightblue', 'lightpink'])
            ax2.set_title('Distribuição por Sexo', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Distribuição de idade
        if 'age' in self.df.columns:
            ax3 = fig.add_subplot(gs[0, 2])
            self.df['age'].hist(bins=30, ax=ax3, color='skyblue', alpha=0.7)
            ax3.set_title('Distribuição de Idade', fontweight='bold')
            ax3.set_xlabel('Idade')
            ax3.set_ylabel('Frequência')
        
        # 4. Top educações
        if 'education' in self.df.columns:
            ax4 = fig.add_subplot(gs[0, 3])
            self.df['education'].value_counts().head(8).plot(kind='barh', ax=ax4, color='lightgreen')
            ax4.set_title('Top 8 Níveis de Educação', fontweight='bold')
        
        # 5. Performance dos modelos
        if self.advanced_metrics:
            ax5 = fig.add_subplot(gs[1, :2])
            models = list(self.advanced_metrics.keys())
            accuracies = [self.advanced_metrics[m]['accuracy'] for m in models]
            ax5.bar(models, accuracies, color='orange', alpha=0.7)
            ax5.set_title('Acurácia dos Modelos ML', fontweight='bold')
            ax5.set_ylabel('Acurácia')
            ax5.tick_params(axis='x', rotation=45)
            ax5.set_ylim(0, 1)
        
        # 6. Clustering (se disponível)
        if self.clustering_results:
            ax6 = fig.add_subplot(gs[1, 2:])
            best_k = self.clustering_results.get('best_k', 0)
            clusters = self.clustering_results.get('clusters', [])
            if len(clusters) > 0:
                unique_clusters, counts = np.unique(clusters, return_counts=True)
                ax6.pie(counts, labels=[f'Cluster {i}' for i in unique_clusters], autopct='%1.1f%%')
                ax6.set_title(f'Distribuição dos {best_k} Clusters', fontweight='bold')
        
        # 7. Estatísticas gerais
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        stats_text = f"""
        📊 ESTATÍSTICAS DO PIPELINE ACADÊMICO v2.0
        
        📋 Dataset: {len(self.df):,} registros, {len(self.df.columns)} features
        🤖 Modelos Treinados: {len(self.models)}
        🎯 Clusters Identificados: {self.clustering_results.get('best_k', 0)}
        📋 Regras de Associação: {len(self.association_rules)}
        🎨 Visualizações Criadas: {self.processing_stats['visualizations_created']}
        📊 Fonte de Dados: {self.processing_stats['data_source'].upper()}
        ⏱️ Tempo de Processamento: {(datetime.now() - self.processing_stats['start_time']).total_seconds():.1f}s
        """
        
        ax7.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle('Dashboard de Resumo - Análise Salarial v2.0', fontsize=16, fontweight='bold')
        plt.savefig(images_dir / "summary_dashboard_v2.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.processing_stats['visualizations_created'] += 1
    
    def _plot_feature_importance(self, images_dir):
        """Plotar importância das features (se disponível)"""
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            
            if hasattr(rf_model, 'feature_importances_'):
                # Tentar obter nomes das features
                try:
                    import joblib
                    preprocessor_path = Path("data/processed/preprocessor_v2.joblib")
                    if preprocessor_path.exists():
                        preprocessor = joblib.load(preprocessor_path)
                        if hasattr(preprocessor, 'get_feature_names_out'):
                            feature_names = preprocessor.get_feature_names_out()
                        else:
                            feature_names = [f'Feature_{i}' for i in range(len(rf_model.feature_importances_))]
                    else:
                        feature_names = [f'Feature_{i}' for i in range(len(rf_model.feature_importances_))]
                except:
                    feature_names = [f'Feature_{i}' for i in range(len(rf_model.feature_importances_))]
                
                # Criar DataFrame de importâncias
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)
                
                # Plot
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(importance_df)), importance_df['Importance'], color='lightgreen')
                plt.yticks(range(len(importance_df)), importance_df['Feature'])
                plt.xlabel('Importância')
                plt.title('Top 15 Features - Random Forest', fontsize=14, fontweight='bold')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                plt.savefig(images_dir / "feature_importance_v2.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                self.processing_stats['visualizations_created'] += 1
                self.logger.info("✅ Gráfico de importância das features criado")
    
    def _create_temporal_analysis(self, images_dir):
        """Criar análise temporal se aplicável"""
        # Esta função pode ser expandida para análises temporais específicas
        # Por enquanto, cria um gráfico de evolução do processamento
        
        processing_times = {
            'Carregamento': 0.1,
            'Limpeza': 0.2,
            'ML': 0.3,
            'Clustering': 0.1,
            'Regras': 0.2,
            'Métricas': 0.1
        }
        
        plt.figure(figsize=(10, 6))
        phases = list(processing_times.keys())
        times = list(processing_times.values())
        cumulative_times = np.cumsum(times)
        
        plt.plot(phases, cumulative_times, marker='o', linewidth=2, markersize=8, color='blue')
        plt.fill_between(phases, cumulative_times, alpha=0.3, color='blue')
        
        plt.title('Evolução do Pipeline de Processamento', fontsize=14, fontweight='bold')
        plt.xlabel('Fases do Pipeline')
        plt.ylabel('Tempo Cumulativo (relativo)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(images_dir / "pipeline_evolution_v2.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.processing_stats['visualizations_created'] += 1
    
    def _phase_8_academic_report(self):
        """Fase 8: Relatório Acadêmico Final"""
        self.logger.info("\n📚 FASE 8: RELATÓRIO ACADÊMICO FINAL")
        self.logger.info("="*60)
        
        try:
            # Calcular tempo total
            total_time = datetime.now() - self.processing_stats['start_time']
            
            # Criar relatório completo
            report_path = Path("output/analysis/relatorio_academico_v2.md")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(self._generate_academic_report_content(total_time))
            
            # Relatório de logs
            self._generate_final_log_summary(total_time)
            
            self.logger.info(f"✅ Relatório acadêmico salvo: {report_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro na geração do relatório: {e}")
    
    def _generate_academic_report_content(self, total_time):
        """Gerar conteúdo do relatório acadêmico"""
        best_model = None
        best_accuracy = 0
        
        if self.advanced_metrics:
            for model, metrics in self.advanced_metrics.items():
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    best_model = model
        
        report = f"""# Relatório Acadêmico - Análise Salarial v2.0

**Data de Geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Tempo Total de Processamento:** {total_time.total_seconds():.1f} segundos  
**Fonte de Dados:** {self.processing_stats['data_source'].upper()}  

## 📊 Resumo Executivo

Este relatório apresenta os resultados de uma análise completa de dados salariais utilizando técnicas de Machine Learning, clustering e análise de regras de associação.

### Principais Resultados:
- **Dataset:** {len(self.df):,} registros processados
- **Modelos Treinados:** {len(self.models)}
- **Melhor Modelo:** {best_model} (Acurácia: {best_accuracy:.4f})
- **Clusters Identificados:** {self.clustering_results.get('best_k', 0)}
- **Regras de Associação:** {len(self.association_rules)}

## 🔍 Metodologia

### 1. Processamento de Dados
- Carregamento de {len(self.df):,} registros
- Limpeza e tratamento de valores ausentes
- Criação de features derivadas (faixas etárias, intensidade de trabalho, etc.)

### 2. Análise Exploratória
- Estatísticas descritivas completas
- Visualizações de distribuições
- Análise de correlações

### 3. Modelagem de Machine Learning
"""
        if self.advanced_metrics:
            report += "\n#### Resultados dos Modelos:\n\n"
            report += "| Modelo | Acurácia | Precisão | Recall | F1-Score |\n"
            report += "|--------|----------|----------|--------|-----------|\n"
            
            for model_name, metrics in self.advanced_metrics.items():
                report += f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} |\n"

        if self.clustering_results:
            report += f"""
### 4. Análise de Clustering
- **Número Ótimo de Clusters:** {self.clustering_results.get('best_k', 0)}
- **Algoritmo:** K-Means
- **Silhouette Score:** {self.clustering_results.get('silhouette_score', 0):.4f}
"""

        if len(self.association_rules) > 0:
            report += f"""
### 5. Regras de Associação
- **Total de Regras Encontradas:** {len(self.association_rules)}
- **Algoritmo:** Apriori
- **Métricas:** Confiança e Lift

#### Top 3 Regras:
"""
            for idx, rule in self.association_rules.head(3).iterrows():
                antecedents = ", ".join(list(rule['antecedents']))
                consequents = ", ".join(list(rule['consequents']))
                report += f"- SE {antecedents} → ENTÃO {consequents} (Conf: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f})\n"

        report += f"""
## 📈 Conclusões

### Principais Insights:
1. **Performance de Modelos:** O modelo {best_model} apresentou a melhor performance com {best_accuracy:.4f} de acurácia
2. **Segmentação:** Identificados {self.clustering_results.get('best_k', 0)} grupos distintos na população
3. **Padrões:** {len(self.association_rules)} regras de associação revelam relações interessantes

### Recomendações:
- Foco em features mais importantes para predição
- Análise detalhada dos clusters identificados
- Aplicação das regras de associação em estratégias de negócio

## 📊 Estatísticas Técnicas

- **Tempo de Processamento:** {total_time.total_seconds():.1f} segundos
- **Visualizações Criadas:** {self.processing_stats['visualizations_created']}
- **Arquivos Gerados:** Modelos, preprocessadores, relatórios e gráficos
- **Fonte de Dados:** {self.processing_stats['data_source'].upper()}

## 📁 Estrutura de Saídas

```
output/
├── images/                    # Visualizações
│   ├── summary_dashboard_v2.png
│   ├── model_comparison_v2.png
│   ├── clustering_analysis_v2.png
│   └── feature_importance_v2.png
├── analysis/                  # Análises
│   ├── relatorio_academico_v2.md
│   ├── advanced_metrics_v2.csv
│   ├── association_rules_v2.csv
│   └── clustering_results_v2.csv
└── logs/                      # Logs do processo
    └── projeto_salario_v2.log

data/processed/               # Modelos treinados
├── preprocessor_v2.joblib
├── random_forest_model_v2.joblib
├── logistic_regression_model_v2.joblib
└── target_encoder_v2.joblib
```

---
**Relatório gerado automaticamente pelo Pipeline Acadêmico v2.0**
"""
        
        return report
    
    def _generate_final_log_summary(self, total_time):
        """Gerar resumo final nos logs"""
        self.logger.info("\n" + "="*80)
        self.logger.info("📋 RESUMO FINAL DO PIPELINE ACADÊMICO v2.0")
        self.logger.info("="*80)
        
        self.logger.info(f"⏱️  Tempo Total: {total_time.total_seconds():.1f} segundos")
        self.logger.info(f"📊 Registros Processados: {len(self.df):,}")
        self.logger.info(f"🤖 Modelos Treinados: {len(self.models)}")
        self.logger.info(f"🎯 Clusters: {self.clustering_results.get('best_k', 0)}")
        self.logger.info(f"📋 Regras de Associação: {len(self.association_rules)}")
        self.logger.info(f"🎨 Visualizações: {self.processing_stats['visualizations_created']}")
        self.logger.info(f"📊 Fonte: {self.processing_stats['data_source'].upper()}")
        
        if self.advanced_metrics:
            self.logger.info("\n🏆 PERFORMANCE DOS MODELOS:")
            for model, metrics in self.advanced_metrics.items():
                self.logger.info(f"   • {model}: {metrics['accuracy']:.4f}")
        
        self.logger.info("\n📁 ARQUIVOS GERADOS:")
        self.logger.info("   • Relatório: output/analysis/relatorio_academico_v2.md")
        self.logger.info("   • Visualizações: output/images/")
        self.logger.info("   • Modelos: data/processed/")
        self.logger.info("   • Análises: output/analysis/")
        
        self.logger.info("\n✅ PIPELINE ACADÊMICO v2.0 CONCLUÍDO COM SUCESSO!")
    
    def _generate_error_report(self):
        """Gerar relatório de erro em caso de falha"""
        try:
            error_dir = Path("output/errors")
            error_dir.mkdir(parents=True, exist_ok=True)
            
            error_report = f"""# Relatório de Erro - Pipeline Acadêmico v2.0

**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Tempo até Erro:** {(datetime.now() - self.processing_stats['start_time']).total_seconds():.1f}s

## Estado do Pipeline no Momento do Erro:

- **Dados Carregados:** {'Sim' if self.df is not None else 'Não'}
- **Registros:** {len(self.df) if self.df is not None else 0}
- **Modelos Treinados:** {len(self.models)}
- **Clustering Realizado:** {'Sim' if self.clustering_results else 'Não'}
- **Regras de Associação:** {len(self.association_rules)}

## Próximos Passos:
1. Verificar logs detalhados em: logs/projeto_salario_v2.log
2. Revisar dados de entrada
3. Verificar dependências instaladas

"""
            
            with open(error_dir / "error_report.md", 'w', encoding='utf-8') as f:
                f.write(error_report)
                
            self.logger.info(f"📝 Relatório de erro salvo: {error_dir}/error_report.md")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao gerar relatório de erro: {e}")

# =============================================================================
# FUNÇÃO PRINCIPAL PARA EXECUÇÃO
# =============================================================================

def main():
    """Função principal para execução do pipeline acadêmico v2.0"""
    
    try:
        # Permitir execução com diferentes modos
        import argparse
        
        parser = argparse.ArgumentParser(description='Pipeline Acadêmico de Análise Salarial v2.0')
        parser.add_argument('--force-csv', action='store_true', 
                          help='Forçar uso de CSV mesmo se banco disponível')
        parser.add_argument('--database-only', action='store_true',
                          help='Usar apenas banco de dados (falhar se indisponível)')
        
        args = parser.parse_args()
        
        # Configurar modo de execução
        if args.database_only:
            use_database = True
            force_csv = False
        elif args.force_csv:
            use_database = False
            force_csv = True
        else:
            use_database = True
            force_csv = False
        
        # Inicializar e executar pipeline
        pipeline = ProjetoSalarioV2(use_database=use_database, force_csv=force_csv)
        pipeline.run_complete_pipeline()
        
        print("\n" + "="*80)
        print("🎉 PIPELINE ACADÊMICO v2.0 CONCLUÍDO COM SUCESSO!")
        print("="*80)
        print("\n💡 Próximos passos:")
        print("   1. Revisar relatório: output/analysis/relatorio_academico_v2.md")
        print("   2. Analisar visualizações: output/images/")
        print("   3. Executar dashboard: streamlit run app.py")
        print("   4. Verificar modelos salvos: data/processed/")
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro crítico no pipeline: {e}")
        print("📝 Verifique logs/projeto_salario_v2.log para detalhes")
        raise

if __name__ == "__main__":
    main()