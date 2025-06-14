#!/usr/bin/env python3
"""
🚀 Pipeline Principal - Sistema Híbrido SQL→CSV
Resultados completos no terminal (como no app.py)
"""

import os
import sys
import logging
import argparse
import traceback
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

def setup_logging():
    """Configurar sistema de logging otimizado"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("HybridPipeline")
    logger.info("🚀 PIPELINE HÍBRIDO SQL→CSV INICIADO")
    logger.info(f"📝 Log salvo em: {log_file}")
    return logger

class HybridPipelineSQL:
    """Pipeline Híbrido com Relatórios Completos no Terminal"""
    
    def __init__(self, force_csv=False, log_level="INFO", show_results=True):
        """Inicializar pipeline híbrido"""
        self.logger = setup_logging()
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        self.force_csv = force_csv
        self.show_results = show_results
        self.data_source = None
        self.results = {}
        self.models = {}
        self.df = None
        self.start_time = datetime.now()
        
        # Métricas de performance
        self.performance_metrics = {
            'start_time': self.start_time,
            'data_source': None,
            'data_load_time': None,
            'ml_training_time': None,
            'total_time': None,
            'records_processed': 0
        }
        
        self.logger.info(f"🔧 Pipeline híbrido inicializado:")
        self.logger.info(f"   • Forçar CSV: {force_csv}")
        self.logger.info(f"   • Nível de log: {log_level}")
        self.logger.info(f"   • Mostrar resultados: {show_results}")
        
        self._initialize_components()

    def _initialize_components(self):
        """Inicializar componentes com fallback"""
        try:
            # 1. Tentar pipeline SQL primeiro (se não forçar CSV)
            if not self.force_csv:
                try:
                    from src.pipelines.data_pipeline import DataPipelineSQL
                    self.sql_pipeline = DataPipelineSQL(force_migration=False)
                    self.logger.info("✅ Pipeline SQL disponível")
                except ImportError as e:
                    self.logger.warning(f"⚠️ Pipeline SQL indisponível: {e}")
                    self.sql_pipeline = None
            else:
                self.sql_pipeline = None
                self.logger.info("📋 Modo CSV forçado")
            
            # 2. Pipeline ML (sempre disponível)
            try:
                from src.pipelines.ml_pipeline import MLPipeline
                self.ml_pipeline = MLPipeline()
                self.logger.info("✅ Pipeline ML inicializado")
            except ImportError as e:
                self.logger.error(f"❌ Pipeline ML não disponível: {e}")
                self.ml_pipeline = None
            
            # 3. Pipelines opcionais
            self._initialize_optional_pipelines()
            
        except Exception as e:
            self.logger.error(f"❌ Erro na inicialização: {e}")

    def _initialize_optional_pipelines(self):
        """Inicializar pipelines opcionais"""
        # Clustering
        try:
            from src.pipelines.clustering_pipeline import ClusteringPipeline
            self.clustering_pipeline = ClusteringPipeline()
            self.logger.info("✅ Pipeline Clustering disponível")
        except ImportError:
            self.clustering_pipeline = None
            self.logger.info("ℹ️ Pipeline Clustering não disponível")
        
        # Association Rules
        try:
            from src.pipelines.association_pipeline import AssociationPipeline
            self.association_pipeline = AssociationPipeline()
            self.logger.info("✅ Pipeline Association disponível")
        except ImportError:
            self.association_pipeline = None
            self.logger.info("ℹ️ Pipeline Association não disponível")

    def run(self) -> Dict[str, Any]:
        """Executar pipeline completo com relatórios no terminal"""
        try:
            self.logger.info("🚀 INICIANDO PIPELINE HÍBRIDO")
            self.logger.info("=" * 60)
            
            # 1. Carregar dados (SQL → CSV fallback)
            self._run_data_pipeline()
            
            if self.df is None or len(self.df) == 0:
                raise ValueError("❌ Nenhum dado foi carregado")
            
            # 2. Machine Learning
            self._run_ml_pipeline()
            
            # 3. Análises opcionais
            self._run_optional_analysis()
            
            # 4. Finalizar
            self._finalize_pipeline()
            
            # 5. 🎯 EXIBIR RESULTADOS COMPLETOS (como app.py)
            if self.show_results:
                self._display_complete_results()
            
            return self._prepare_results()
            
        except Exception as e:
            self.logger.error(f"❌ Erro crítico no pipeline: {e}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e), 'data_source': self.data_source}

    def _run_data_pipeline(self):
        """Executar carregamento de dados com fallback SQL→CSV"""
        self.logger.info("📊 CARREGAMENTO DE DADOS")
        self.logger.info("-" * 40)
        
        data_start = datetime.now()
        
        # Tentar SQL primeiro (se disponível)
        if self.sql_pipeline and not self.force_csv:
            self.logger.info("🗄️ Tentando carregar dados do SQL...")
            try:
                self.df = self.sql_pipeline.run()
                if self.df is not None and len(self.df) > 0:
                    self.data_source = 'sql'
                    self.logger.info(f"✅ Dados carregados do SQL: {len(self.df):,} registros")
                    self._log_data_details()
                else:
                    self.logger.warning("⚠️ SQL retornou dados vazios, tentando CSV...")
                    self._load_from_csv()
            except Exception as e:
                self.logger.warning(f"⚠️ Erro no SQL: {e}")
                self.logger.info("🔄 Fazendo fallback para CSV...")
                self._load_from_csv()
        else:
            # Carregar diretamente do CSV
            self._load_from_csv()
        
        # Calcular tempo de carregamento
        data_time = datetime.now() - data_start
        self.performance_metrics['data_load_time'] = data_time.total_seconds()
        self.performance_metrics['data_source'] = self.data_source
        self.performance_metrics['records_processed'] = len(self.df) if self.df is not None else 0
        
        self.logger.info(f"⏱️ Tempo de carregamento: {data_time.total_seconds():.2f}s")

    def _load_from_csv(self):
        """Carregar dados do arquivo CSV"""
        csv_paths = [
            "data/raw/4-Carateristicas_salario.csv",
            "4-Carateristicas_salario.csv",
            "bkp/4-Carateristicas_salario.csv",
            "data/4-Carateristicas_salario.csv"
        ]
        
        for csv_path in csv_paths:
            if Path(csv_path).exists():
                try:
                    self.logger.info(f"📋 Carregando CSV: {csv_path}")
                    self.df = pd.read_csv(csv_path)
                    
                    if len(self.df) > 0:
                        self.data_source = 'csv'
                        self.logger.info(f"✅ CSV carregado: {len(self.df):,} registros, {len(self.df.columns)} colunas")
                        
                        # Limpeza básica
                        self._basic_cleaning()
                        self._log_data_details()
                        return
                    
                except Exception as e:
                    self.logger.error(f"❌ Erro ao carregar {csv_path}: {e}")
        
        # Se chegou aqui, não encontrou nenhum CSV
        self.logger.error("❌ Nenhum arquivo CSV encontrado!")
        self.logger.info("💡 Locais procurados:")
        for path in csv_paths:
            self.logger.info(f"   • {path}")

    def _basic_cleaning(self):
        """Limpeza básica dos dados CSV"""
        if self.df is None:
            return
        
        initial_size = len(self.df)
        
        # Remover espaços e caracteres especiais
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].astype(str).str.strip()
        
        # Substituir '?' por NaN
        self.df = self.df.replace('?', np.nan)
        
        # Remover linhas completamente vazias
        self.df = self.df.dropna(how='all')
        
        final_size = len(self.df)
        if final_size != initial_size:
            self.logger.info(f"🧹 Limpeza: {initial_size:,} → {final_size:,} registros")

    def _log_data_details(self):
        """Log detalhado dos dados carregados"""
        if self.df is None:
            return
        
        self.logger.info("📈 DETALHES DOS DADOS:")
        self.logger.info(f"   📋 Registros: {len(self.df):,}")
        self.logger.info(f"   📊 Colunas: {len(self.df.columns)}")
        self.logger.info(f"   💾 Memória: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Colunas por tipo
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        self.logger.info(f"   🔢 Numéricas: {len(numeric_cols)}")
        self.logger.info(f"   📝 Categóricas: {len(categorical_cols)}")
        
        # Dados ausentes
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            self.logger.warning("⚠️ DADOS AUSENTES:")
            for col, missing_count in missing_data[missing_data > 0].items():
                percentage = (missing_count / len(self.df)) * 100
                self.logger.warning(f"   • {col}: {missing_count} ({percentage:.1f}%)")
        else:
            self.logger.info("✅ Sem dados ausentes")
        
        # Duplicatas
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.logger.warning(f"⚠️ {duplicates} duplicatas ({duplicates/len(self.df)*100:.1f}%)")
        else:
            self.logger.info("✅ Sem duplicatas")
        
        # Distribuição target (se existir)
        if 'salary' in self.df.columns:
            target_dist = self.df['salary'].value_counts()
            self.logger.info("🎯 DISTRIBUIÇÃO TARGET:")
            for value, count in target_dist.items():
                percentage = (count / len(self.df)) * 100
                self.logger.info(f"   • {value}: {count:,} ({percentage:.1f}%)")

    def _run_ml_pipeline(self):
        """Executar pipeline de ML"""
        if not self.ml_pipeline:
            self.logger.warning("⚠️ Pipeline ML não disponível")
            return
        
        self.logger.info("🤖 MACHINE LEARNING")
        self.logger.info("-" * 40)
        
        ml_start = datetime.now()
        
        try:
            self.models, ml_results = self.ml_pipeline.run(self.df)
            
            if self.models:
                self.logger.info(f"✅ {len(self.models)} modelos treinados")
                
                # Log performance de cada modelo
                for name, model_info in self.models.items():
                    if isinstance(model_info, dict) and 'accuracy' in model_info:
                        accuracy = model_info['accuracy']
                        self.logger.info(f"   • {name}: {accuracy:.4f}")
                        
                        # Classificação de performance
                        if accuracy > 0.90:
                            self.logger.info(f"     🏆 EXCELENTE")
                        elif accuracy > 0.85:
                            self.logger.info(f"     ✅ MUITO BOA")
                        elif accuracy > 0.80:
                            self.logger.info(f"     ⚠️ BOA")
                        else:
                            self.logger.warning(f"     ❌ REGULAR")
                
                # Identificar melhor modelo
                best_model, best_score = self._find_best_model()
                if best_model:
                    self.logger.info(f"🏆 MELHOR MODELO: {best_model} ({best_score:.4f})")
                
                self.results['ml_results'] = ml_results
            else:
                self.logger.warning("⚠️ Nenhum modelo foi treinado")
        
        except Exception as e:
            self.logger.error(f"❌ Erro no ML: {e}")
            self.logger.error(traceback.format_exc())
        
        ml_time = datetime.now() - ml_start
        self.performance_metrics['ml_training_time'] = ml_time.total_seconds()
        self.logger.info(f"⏱️ Tempo ML: {ml_time.total_seconds():.2f}s")

    def _find_best_model(self) -> Tuple[Optional[str], float]:
        """Encontrar melhor modelo"""
        best_model = None
        best_score = 0.0
        
        for model_name, model_info in self.models.items():
            if isinstance(model_info, dict) and 'accuracy' in model_info:
                accuracy = model_info['accuracy']
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model_name
        
        return best_model, best_score

    def _run_optional_analysis(self):
        """Executar análises opcionais"""
        self.logger.info("📊 ANÁLISES OPCIONAIS")
        self.logger.info("-" * 40)
        
        # Clustering
        if self.clustering_pipeline:
            try:
                self.logger.info("🎯 Executando clustering...")
                clustering_results = self.clustering_pipeline.run(self.df)
                if clustering_results:
                    self.results['clustering'] = clustering_results
                    self.logger.info("✅ Clustering concluído")
                    
                    # Log resultados de clustering
                    for algorithm, result in clustering_results.items():
                        if isinstance(result, dict):
                            if 'n_clusters' in result:
                                self.logger.info(f"   • {algorithm}: {result['n_clusters']} clusters")
                            if 'silhouette_score' in result:
                                score = result['silhouette_score']
                                self.logger.info(f"     Silhouette: {score:.4f}")
                else:
                    self.logger.warning("⚠️ Clustering não retornou resultados")
            except Exception as e:
                self.logger.warning(f"⚠️ Erro no clustering: {e}")
        else:
            self.logger.info("ℹ️ Clustering não disponível")
        
        # Association Rules
        if self.association_pipeline:
            try:
                self.logger.info("📋 Executando regras de associação...")
                association_results = self.association_pipeline.run(self.df)
                if association_results:
                    self.results['association_rules'] = association_results
                    self.logger.info("✅ Regras de associação concluídas")
                    
                    # Log resultados
                    if 'rules' in association_results:
                        rules_count = len(association_results['rules'])
                        self.logger.info(f"   • {rules_count} regras encontradas")
                else:
                    self.logger.warning("⚠️ Regras de associação não retornaram resultados")
            except Exception as e:
                self.logger.warning(f"⚠️ Erro nas regras de associação: {e}")
        else:
            self.logger.info("ℹ️ Regras de associação não disponíveis")

    def _finalize_pipeline(self):
        """Finalizar pipeline"""
        total_time = datetime.now() - self.start_time
        self.performance_metrics['total_time'] = total_time.total_seconds()
        
        self.logger.info("🎉 PIPELINE CONCLUÍDO")
        self.logger.info("=" * 60)
        self.logger.info(f"⏱️ Tempo total: {total_time.total_seconds():.2f}s")
        self.logger.info(f"📊 Fonte dos dados: {self.data_source.upper()}")
        self.logger.info(f"📋 Registros processados: {self.performance_metrics['records_processed']:,}")
        
        if self.models:
            self.logger.info(f"🤖 Modelos treinados: {len(self.models)}")
            best_model, best_score = self._find_best_model()
            if best_model:
                self.logger.info(f"🏆 Melhor performance: {best_model} ({best_score:.4f})")
        
        # Salvar resultados
        self._save_results()

    def _display_complete_results(self):
        """🎯 EXIBIR RESULTADOS COMPLETOS NO TERMINAL - Como app.py"""
        print("\n" + "🎯" * 80)
        print("🎯 RELATÓRIO COMPLETO DE RESULTADOS - ANÁLISE DE SALÁRIOS")
        print("🎯" * 80)
        
        # 1. OVERVIEW DOS DADOS
        self._display_data_overview()
        
        # 2. ANÁLISE EXPLORATÓRIA
        self._display_eda_results()
        
        # 3. MACHINE LEARNING
        self._display_ml_results()
        
        # 4. MÉTRICAS DE PERFORMANCE
        self._display_performance_metrics()
        
        # 5. INSIGHTS E CONCLUSÕES
        self._display_insights()
        
        print("\n" + "🎯" * 80)
        print("✅ RELATÓRIO COMPLETO EXIBIDO COM SUCESSO!")
        print("🎯" * 80)

    def _display_data_overview(self):
        """Exibir overview dos dados"""
        print(f"\n📊 1. OVERVIEW DOS DADOS")
        print("-" * 50)
        
        if self.df is not None:
            print(f"📋 Total de Registros: {len(self.df):,}")
            print(f"📊 Total de Colunas: {len(self.df.columns)}")
            print(f"💾 Tamanho em Memória: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"🗄️ Fonte dos Dados: {self.data_source.upper()}")
            
            # Tipos de dados
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            
            print(f"\n📈 TIPOS DE VARIÁVEIS:")
            print(f"   🔢 Numéricas: {len(numeric_cols)} ({', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''})")
            print(f"   📝 Categóricas: {len(categorical_cols)} ({', '.join(categorical_cols[:3])}{'...' if len(categorical_cols) > 3 else ''})")
            
            # Qualidade dos dados
            missing_data = self.df.isnull().sum()
            duplicates = self.df.duplicated().sum()
            
            print(f"\n🧹 QUALIDADE DOS DADOS:")
            if missing_data.sum() > 0:
                cols_with_missing = missing_data[missing_data > 0]
                print(f"   ⚠️ Dados Ausentes: {missing_data.sum():,} valores em {len(cols_with_missing)} colunas")
                for col, count in cols_with_missing.head(3).items():
                    percentage = (count / len(self.df)) * 100
                    print(f"      • {col}: {count:,} ({percentage:.1f}%)")
            else:
                print(f"   ✅ Dados Ausentes: Nenhum")
            
            if duplicates > 0:
                print(f"   ⚠️ Duplicatas: {duplicates:,} ({duplicates/len(self.df)*100:.1f}%)")
            else:
                print(f"   ✅ Duplicatas: Nenhuma")

    def _display_eda_results(self):
        """Exibir resultados da análise exploratória"""
        print(f"\n🔍 2. ANÁLISE EXPLORATÓRIA")
        print("-" * 50)
        
        if self.df is not None and 'salary' in self.df.columns:
            # Distribuição da variável target
            target_dist = self.df['salary'].value_counts()
            target_dist_pct = self.df['salary'].value_counts(normalize=True)
            
            print(f"🎯 DISTRIBUIÇÃO DA VARIÁVEL TARGET (salary):")
            for value, count in target_dist.items():
                percentage = target_dist_pct[value] * 100
                bar_length = int(percentage / 2)  # Escala para visualização
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"   {value:<8}: {count:>8,} ({percentage:>5.1f}%) {bar}")
            
            # Estatísticas das variáveis numéricas
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"\n📊 ESTATÍSTICAS DAS VARIÁVEIS NUMÉRICAS:")
                stats = self.df[numeric_cols].describe()
                
                for col in numeric_cols[:4]:  # Mostrar apenas as primeiras 4
                    if col in stats.columns:
                        mean_val = stats.loc['mean', col]
                        std_val = stats.loc['std', col]
                        min_val = stats.loc['min', col]
                        max_val = stats.loc['max', col]
                        
                        print(f"   {col:<15}: Média={mean_val:>8.1f} | Desvio={std_val:>8.1f} | Min={min_val:>8.0f} | Max={max_val:>8.0f}")
            
            # Top categorias das variáveis categóricas
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            important_cats = ['workclass', 'education', 'occupation', 'sex']
            
            print(f"\n📝 TOP CATEGORIAS DAS PRINCIPAIS VARIÁVEIS:")
            for col in important_cats:
                if col in categorical_cols and col in self.df.columns:
                    top_values = self.df[col].value_counts().head(3)
                    print(f"   {col:<12}: {dict(top_values)}")

    def _display_ml_results(self):
        """Exibir resultados de Machine Learning"""
        print(f"\n🤖 3. RESULTADOS DE MACHINE LEARNING")
        print("-" * 50)
        
        if self.models and len(self.models) > 0:
            print(f"✅ Modelos Treinados: {len(self.models)}")
            
            # Extrair métricas dos modelos
            model_metrics = {}
            for name, model_info in self.models.items():
                if isinstance(model_info, dict):
                    model_metrics[name] = model_info
                else:
                    # Se for o objeto do modelo diretamente, pegar dos resultados
                    if name in self.results.get('ml_results', {}):
                        model_metrics[name] = self.results['ml_results'][name]
            
            # Exibir performance de cada modelo
            print(f"\n📈 PERFORMANCE DOS MODELOS:")
            sorted_models = []
            
            for name, metrics in model_metrics.items():
                if 'accuracy' in metrics:
                    accuracy = metrics['accuracy']
                    sorted_models.append((name, accuracy, metrics))
            
            # Ordenar por accuracy
            sorted_models.sort(key=lambda x: x[1], reverse=True)
            
            for i, (name, accuracy, metrics) in enumerate(sorted_models, 1):
                # Barra de progresso visual
                bar_length = int(accuracy * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                
                # Classificação de performance
                if accuracy > 0.90:
                    grade = "🏆 EXCELENTE"
                elif accuracy > 0.85:
                    grade = "✅ MUITO BOM"
                elif accuracy > 0.80:
                    grade = "⚠️ BOM"
                else:
                    grade = "❓ REGULAR"
                
                print(f"   {i}. {name:<20}: {accuracy:.4f} ({accuracy*100:5.1f}%) {bar} {grade}")
                
                # Métricas adicionais se disponíveis
                additional_metrics = []
                for metric in ['precision', 'recall', 'f1_score', 'roc_auc']:
                    if metric in metrics:
                        additional_metrics.append(f"{metric}={metrics[metric]:.3f}")
                
                if additional_metrics:
                    print(f"      📊 {' | '.join(additional_metrics)}")
            
            # Melhor modelo
            if sorted_models:
                best_name, best_accuracy, _ = sorted_models[0]
                print(f"\n🏆 MELHOR MODELO: {best_name}")
                print(f"🎯 ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.1f}%)")
        else:
            print("❌ Nenhum modelo foi treinado com sucesso")

    def _display_performance_metrics(self):
        """Exibir métricas de performance do pipeline"""
        print(f"\n⏱️ 4. MÉTRICAS DE PERFORMANCE DO PIPELINE")
        print("-" * 50)
        
        metrics = self.performance_metrics
        
        print(f"📊 TEMPOS DE EXECUÇÃO:")
        print(f"   ⏱️ Carregamento de dados: {metrics.get('data_load_time', 0):.2f}s")
        print(f"   🤖 Treinamento ML: {metrics.get('ml_training_time', 0):.2f}s")
        print(f"   🏁 Tempo total: {metrics.get('total_time', 0):.2f}s")
        
        print(f"\n📈 PROCESSAMENTO:")
        print(f"   📋 Registros processados: {metrics.get('records_processed', 0):,}")
        print(f"   🗄️ Fonte de dados: {metrics.get('data_source', 'N/A').upper()}")
        
        # Taxa de processamento
        if metrics.get('total_time', 0) > 0 and metrics.get('records_processed', 0) > 0:
            rate = metrics['records_processed'] / metrics['total_time']
            print(f"   🚀 Taxa de processamento: {rate:,.0f} registros/segundo")
        
        # Eficiência de memória
        if self.df is not None:
            memory_mb = self.df.memory_usage(deep=True).sum() / 1024**2
            print(f"   💾 Uso de memória: {memory_mb:.2f} MB")

    def _display_insights(self):
        """Exibir insights e conclusões"""
        print(f"\n💡 5. INSIGHTS E CONCLUSÕES")
        print("-" * 50)
        
        insights = []
        
        # Insights sobre os dados
        if self.df is not None:
            total_records = len(self.df)
            
            if 'salary' in self.df.columns:
                high_salary_pct = (self.df['salary'] == '>50K').mean() * 100
                
                if high_salary_pct < 30:
                    insights.append(f"📊 Apenas {high_salary_pct:.1f}% ganham mais de $50K - dataset desbalanceado")
                
                # Insights por idade
                if 'age' in self.df.columns:
                    avg_age = self.df['age'].mean()
                    insights.append(f"👥 Idade média: {avg_age:.1f} anos")
                
                # Insights por educação
                if 'education-num' in self.df.columns:
                    high_earners_edu = self.df[self.df['salary'] == '>50K']['education-num'].mean()
                    low_earners_edu = self.df[self.df['salary'] == '<=50K']['education-num'].mean()
                    
                    if high_earners_edu > low_earners_edu:
                        diff = high_earners_edu - low_earners_edu
                        insights.append(f"🎓 Quem ganha mais tem {diff:.1f} anos a mais de educação em média")
                
                # Insights por horas trabalhadas
                if 'hours-per-week' in self.df.columns:
                    high_earners_hours = self.df[self.df['salary'] == '>50K']['hours-per-week'].mean()
                    low_earners_hours = self.df[self.df['salary'] == '<=50K']['hours-per-week'].mean()
                    
                    if high_earners_hours > low_earners_hours:
                        diff = high_earners_hours - low_earners_hours
                        insights.append(f"⏰ Quem ganha mais trabalha {diff:.1f} horas/semana a mais")
        
        # Insights sobre modelos
        if self.models and len(self.models) > 0:
            best_model, best_score = self._find_best_model()
            if best_model and best_score > 0.80:
                insights.append(f"🤖 {best_model} alcançou {best_score:.1%} de accuracy - bom desempenho")
            elif best_model:
                insights.append(f"🤖 {best_model} alcançou {best_score:.1%} de accuracy - necessita melhorias")
        
        # Insights sobre performance
        if self.performance_metrics.get('total_time', 0) < 5:
            insights.append(f"⚡ Pipeline executou em {self.performance_metrics['total_time']:.1f}s - muito eficiente")
        
        # Exibir insights
        if insights:
            for i, insight in enumerate(insights, 1):
                print(f"   {i}. {insight}")
        else:
            print("   ℹ️ Análise concluída - execute com mais dados para insights detalhados")
        
        # Recomendações
        print(f"\n🎯 PRÓXIMOS PASSOS RECOMENDADOS:")
        recommendations = [
            "📊 Execute o dashboard: streamlit run app.py",
            "🔍 Analise os gráficos interativos no Streamlit",
            "🤖 Teste diferentes algoritmos de ML",
            "📈 Colete mais dados para melhorar os modelos"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

    def _save_results(self):
        """Salvar resultados do pipeline"""
        try:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            # Salvar estado do pipeline
            pipeline_state = {
                'data_source': self.data_source,
                'models_count': len(self.models),
                'performance_metrics': self.performance_metrics,
                'results_summary': {
                    'has_ml': len(self.models) > 0,
                    'has_clustering': 'clustering' in self.results,
                    'has_association': 'association_rules' in self.results
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Salvar em JSON
            state_file = output_dir / "pipeline_state.json"
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(pipeline_state, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"💾 Estado salvo: {state_file}")
            
            # Salvar modelos se existirem
            if self.models:
                models_file = output_dir / "models_summary.json"
                models_summary = {}
                
                for name, model_info in self.models.items():
                    if isinstance(model_info, dict):
                        models_summary[name] = {
                            k: v for k, v in model_info.items() 
                            if isinstance(v, (int, float, str, bool))
                        }
                
                with open(models_file, 'w', encoding='utf-8') as f:
                    json.dump(models_summary, f, indent=2, default=str, ensure_ascii=False)
                
                self.logger.info(f"💾 Modelos salvos: {models_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao salvar resultados: {e}")

    def _prepare_results(self) -> Dict[str, Any]:
        """Preparar resultados finais"""
        return {
            'df': self.df,
            'models': self.models,
            'results': self.results,
            'data_source': self.data_source,
            'performance_metrics': self.performance_metrics,
            'status': self._generate_status_message()
        }

    def _generate_status_message(self) -> str:
        """Gerar mensagem de status final"""
        if self.df is not None and len(self.df) > 0:
            if self.models and len(self.models) > 0:
                best_model, best_score = self._find_best_model()
                return f"✅ Pipeline concluído - Fonte: {self.data_source.upper()} | Melhor modelo: {best_model} ({best_score:.4f})"
            else:
                return f"⚠️ Dados carregados via {self.data_source.upper()}, mas problemas no ML"
        else:
            return "❌ Falha no carregamento de dados"

def setup_database():
    """Configurar banco de dados (placeholder)"""
    print("🗄️ Configuração de Banco de Dados")
    print("=" * 40)
    
    try:
        # Tentar importar e usar setup real
        from src.database.connection import create_connection
        from src.database.migration import run_migration
        
        print("📋 Testando conexão...")
        conn = create_connection()
        if conn:
            print("✅ Conexão com banco OK")
            
            print("🔄 Executando migração...")
            success = run_migration()
            if success:
                print("✅ Migração concluída com sucesso")
            else:
                print("⚠️ Problemas na migração")
            
            conn.close()
        else:
            print("❌ Erro na conexão com banco")
            
    except ImportError:
        print("⚠️ Módulos de banco não encontrados")
        print("💡 Sistema funcionará em modo CSV")
    except Exception as e:
        print(f"❌ Erro na configuração: {e}")
        print("💡 Sistema funcionará em modo CSV")

def main():
    """Função principal com argumentos otimizados"""
    parser = argparse.ArgumentParser(description='Pipeline Híbrido SQL→CSV')
    parser.add_argument('--csv-only', action='store_true', help='Forçar uso apenas de CSV')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--setup-db', action='store_true', help='Configurar banco de dados')
    parser.add_argument('--no-results', action='store_true', help='Não exibir relatório completo')
    
    args = parser.parse_args()
    
    try:
        if args.setup_db:
            setup_database()
            return
        
        # Executar pipeline principal
        pipeline = HybridPipelineSQL(
            force_csv=args.csv_only,
            log_level=args.log_level,
            show_results=not args.no_results
        )
        
        results = pipeline.run()
        
        if 'error' not in results:
            print(f"\n🎉 PIPELINE CONCLUÍDO COM SUCESSO!")
            print(f"📊 Fonte: {results['data_source'].upper()}")
            print(f"📋 Registros: {len(results['df']):,}")
            print(f"🤖 Modelos: {len(results['models'])}")
            print(f"⏱️ Tempo: {results['performance_metrics']['total_time']:.2f}s")
            print(f"\n💡 Próximo passo: streamlit run app.py")
        else:
            print(f"\n❌ ERRO NO PIPELINE: {results['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro crítico: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()