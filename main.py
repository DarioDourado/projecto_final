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
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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
    """Pipeline híbrido com otimização automática para datasets grandes"""
    
    def __init__(self, force_csv=False, log_level="INFO", show_results=True, auto_optimize=True):
        """Inicializar pipeline híbrido com otimização automática"""
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        # NOVO: Configurações de otimização
        self.auto_optimize = auto_optimize
        self.optimization_threshold = 50000  # Se > 50k registros, otimizar
        self.sample_size = 15000  # Tamanho da amostra otimizada
        self.is_optimized_run = False
        self.original_size = 0
        
        # Configurações básicas
        self.force_csv = force_csv
        self.show_results = show_results
        self.df = None
        self.results = {}
        self.models = {}
        self.performance_metrics = {}
        
        # Configurar pipelines
        self._setup_pipelines()
        
        # Auto-detectar necessidade de otimização
        if self.auto_optimize:
            self._check_optimization_needed()
    
    def setup_logging(self, log_level="INFO"):
        """Configurar sistema de logging"""
        # Criar diretório de logs se não existir
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configurar formato
        log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        
        # Configurar logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        
        # Suprimir logs desnecessários
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('seaborn').setLevel(logging.WARNING)
        logging.getLogger('sklearn').setLevel(logging.WARNING)
    
    def _setup_pipelines(self):
        """Configurar todos os pipelines"""
        self.logger.info("🔧 Configurando pipelines...")
        
        # ML Pipeline
        try:
            from src.pipelines.ml_pipeline import MLPipeline
            self.ml_pipeline = MLPipeline()
            self.logger.info("✅ Pipeline ML disponível")
        except Exception as e:
            self.ml_pipeline = None
            self.logger.warning(f"⚠️ Pipeline ML não disponível: {e}")
        
        # Clustering Pipeline
        try:
            from src.pipelines.clustering_pipeline import ClusteringPipeline
            self.clustering_pipeline = ClusteringPipeline()
            self.logger.info("✅ Pipeline Clustering disponível")
        except Exception as e:
            self.clustering_pipeline = None
            self.logger.warning(f"⚠️ Pipeline Clustering não disponível: {e}")
        
        # Association Rules Pipeline (NOVO)
        try:
            from src.pipelines.association_pipeline import AssociationPipeline
            self.association_pipeline = AssociationPipeline()
            self.logger.info("✅ Pipeline Association Rules disponível")
        except Exception as e:
            self.association_pipeline = None
            self.logger.warning(f"⚠️ Pipeline Association Rules não disponível: {e}")

    def _check_optimization_needed(self):
        """Verificar se otimização é necessária baseada no tamanho do dataset"""
        try:
            # Verificar tamanho do arquivo primeiro
            csv_path = Path("data/raw/4-Carateristicas_salario.csv")
            if csv_path.exists():
                file_size_mb = csv_path.stat().st_size / (1024 * 1024)
                
                # Se arquivo > 10MB, provavelmente precisa otimização
                if file_size_mb > 10:
                    # Contar linhas usando pandas
                    try:
                        # Ler apenas uma linha para verificar se pode ser carregado
                        df_test = pd.read_csv(csv_path, nrows=1)
                        
                        # Estimar número de linhas baseado no tamanho do arquivo
                        estimated_lines = int(file_size_mb * 1000)  # Estimativa rough
                        
                        if estimated_lines > self.optimization_threshold:
                            self.is_optimized_run = True
                            self.original_size = estimated_lines
                            self.logger.info(f"🎯 AUTO-OTIMIZAÇÃO ATIVADA")
                            self.logger.info(f"📊 Dataset estimado: ~{estimated_lines:,} registros")
                            self.logger.info(f"⚡ Modo otimizado: {self.sample_size:,} amostras")
                            return
                            
                    except Exception as e:
                        self.logger.debug(f"Erro na estimativa: {e}")
                
        except Exception as e:
            self.logger.debug(f"Verificação de otimização falhou: {e}")
    
    def load_data(self):
        """Carregar dados com otimização automática"""
        self.logger.info("📊 Carregando dados...")
        
        try:
            csv_path = Path("data/raw/4-Carateristicas_salario.csv")
            
            if not csv_path.exists():
                self.logger.error(f"❌ Arquivo não encontrado: {csv_path}")
                return None
            
            # Se otimização ativada, usar amostragem
            if self.is_optimized_run:
                return self._load_data_optimized(csv_path)
            else:
                # Carregamento normal
                self.df = pd.read_csv(csv_path)
                self.original_size = len(self.df)
                self.logger.info(f"✅ Dados carregados: {len(self.df):,} registros")
                return self.df
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar dados: {e}")
            return None
    
    def _load_data_optimized(self, csv_path):
        """Carregar dados com amostragem inteligente"""
        self.logger.info("⚡ CARREGAMENTO OTIMIZADO")
        
        try:
            # Primeiro, ler uma amostra pequena para entender os dados
            df_peek = pd.read_csv(csv_path, nrows=1000)
            self.logger.info(f"🔍 Análise preliminar: {len(df_peek.columns)} colunas")
            
            # Carregar amostra diretamente
            self.logger.info(f"📊 Carregando amostra de {self.sample_size:,} registros...")
            
            # Usar skip para pegar amostra distribuída
            skip_rows = max(1, self.original_size // self.sample_size)
            
            try:
                # Tentar carregar amostra sistemática
                self.df = pd.read_csv(csv_path, skiprows=lambda i: i > 0 and i % skip_rows != 0)
                
                # Limitar ao tamanho desejado
                if len(self.df) > self.sample_size:
                    self.df = self.df.sample(self.sample_size, random_state=42).reset_index(drop=True)
                
            except Exception:
                # Fallback: carregar primeiras N linhas
                self.df = pd.read_csv(csv_path, nrows=self.sample_size)
            
            self.logger.info(f"✅ Amostra criada: {len(self.df):,} registros")
            
            # Verificar distribuição da variável target se existir
            if 'salary' in self.df.columns:
                dist = self.df['salary'].value_counts(normalize=True)
                self.logger.info(f"🎯 Distribuição: {dict(dist)}")
            
            return self.df
                
        except Exception as e:
            self.logger.error(f"❌ Erro no carregamento otimizado: {e}")
            # Fallback final
            try:
                self.df = pd.read_csv(csv_path, nrows=self.sample_size)
                self.logger.info(f"⚠️ Fallback: primeiros {len(self.df):,} registros")
                return self.df
            except:
                return None
    
    def run(self):
        """Executar pipeline completo"""
        start_time = datetime.now()
        self.logger.info("🚀 INICIANDO PIPELINE HÍBRIDO")
        self.logger.info("=" * 60)
        
        try:
            # 1. Carregar dados - CORREÇÃO DO ERRO
            data_loaded = self.load_data()
            if data_loaded is None or self.df is None or self.df.empty:
                self.logger.error("❌ Falha no carregamento dos dados")
                return None
            
            # 2. Machine Learning
            if self.ml_pipeline:
                try:
                    self.logger.info("🤖 Executando Machine Learning...")
                    models, results = self.ml_pipeline.run(self.df)
                    if models and results:
                        self.results['ml'] = {'models': results, 'trained_models': models}
                        self.models = models
                        self.logger.info("✅ Machine Learning concluído")
                        
                        # Log detalhado dos resultados ML
                        for model_name, metrics in results.items():
                            if isinstance(metrics, dict) and 'accuracy' in metrics:
                                accuracy = metrics['accuracy']
                                self.logger.info(f"   📊 {model_name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
                        
                except Exception as e:
                    self.logger.error(f"❌ Erro no ML: {e}")
            
            # 3. Clustering (com otimização)
            if self.clustering_pipeline:
                try:
                    self.logger.info("🎯 Executando Clustering...")
                    
                    # Otimizar se necessário
                    if self.is_optimized_run:
                        self._optimize_clustering()
                    
                    clustering_results = self.clustering_pipeline.run(self.df)
                    if clustering_results:
                        self.results['clustering'] = clustering_results
                        self.logger.info("✅ Clustering concluído")
                        
                        # Log detalhado dos resultados de clustering
                        for method, result in clustering_results.items():
                            if result and isinstance(result, dict):
                                n_clusters = result.get('n_clusters', 0)
                                silhouette = result.get('silhouette_score', 0)
                                noise_pct = result.get('noise_percentage', 0)
                                
                                self.logger.info(f"   📊 {method}: {n_clusters} clusters, Silhouette={silhouette:.3f}")
                                if noise_pct > 0:
                                    self.logger.info(f"      🔍 Ruído detectado: {noise_pct:.1f}%")
                        
                except Exception as e:
                    self.logger.error(f"❌ Erro no Clustering: {e}")
            elif self.is_optimized_run:
                self.logger.info("🎯 Clustering: PULADO (modo otimizado)")
            
            # 4. Association Rules (pular se otimizado)
            if self.association_pipeline and not self.is_optimized_run:
                try:
                    self.logger.info("📋 Executando Association Rules...")
                    association_results = self.association_pipeline.run(self.df)
                    if association_results:
                        self.results['association_rules'] = association_results
                        self.logger.info("✅ Association Rules concluído")
                        
                        # Log detalhado das regras
                        if 'rules' in association_results:
                            rules_count = len(association_results['rules'])
                            self.logger.info(f"   📊 {rules_count} regras descobertas")
                        
                except Exception as e:
                    self.logger.error(f"❌ Erro no Association Rules: {e}")
            elif self.is_optimized_run:
                self.logger.info("📋 Association Rules: PULADO (modo otimizado)")
            
            # 5. Calcular métricas de performance
            total_time = datetime.now() - start_time
            self.performance_metrics['total_time'] = total_time.total_seconds()
            self.performance_metrics['records_processed'] = len(self.df)
            self.performance_metrics['data_source'] = 'csv'
            
            # 6. Exibir resultados
            if self.show_results:
                self._display_results()
            
            self.logger.info("🎯 PIPELINE CONCLUÍDO COM SUCESSO")
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Erro crítico no pipeline: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
    def _optimize_clustering(self):
        """Otimizar clustering para datasets grandes"""
        if hasattr(self.clustering_pipeline, 'clustering_analysis'):
            self.logger.info("⚡ Aplicando otimizações de clustering...")
            
            # Reduzir número de testes do DBSCAN
            try:
                original_method = self.clustering_pipeline.clustering_analysis.perform_dbscan_analysis
                
                def fast_dbscan(X):
                    """DBSCAN otimizado"""
                    from sklearn.cluster import DBSCAN
                    from sklearn.metrics import silhouette_score
                    from sklearn.preprocessing import StandardScaler
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Apenas configurações promissoras
                    configs = [(0.9, 7), (1.2, 5), (1.3, 6), (0.8, 8)]
                    
                    best_silhouette = -1
                    best_result = None
                    
                    self.logger.info(f"⚡ Testando {len(configs)} configurações DBSCAN...")
                    
                    for i, (eps, min_samples) in enumerate(configs, 1):
                        try:
                            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                            clusters = dbscan.fit_predict(X_scaled)
                            
                            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                            n_noise = list(clusters).count(-1)
                            
                            if n_clusters > 1:
                                mask = clusters != -1
                                if mask.sum() > 10:
                                    silhouette = silhouette_score(X_scaled[mask], clusters[mask])
                                    
                                    self.logger.info(f"   {i}/{len(configs)}: eps={eps}, min_samples={min_samples} → "
                                                   f"Silhouette={silhouette:.3f}")
                                    
                                    if silhouette > best_silhouette:
                                        best_silhouette = silhouette
                                        best_result = (clusters, n_clusters, silhouette)
                        except:
                            continue
                    
                    return best_result if best_result else (None, 0, -1)
                
                # Substituir método temporariamente
                self.clustering_pipeline.clustering_analysis.perform_dbscan_analysis = fast_dbscan
                
            except Exception as e:
                self.logger.debug(f"Erro na otimização DBSCAN: {e}")
    
    def _display_results(self):
        """Exibir resultados no terminal"""
        print("\n" + "🎯" * 80)
        print("🎯 RELATÓRIO COMPLETO DE RESULTADOS - ANÁLISE DE SALÁRIOS")
        print("🎯" * 80)
        
        # 1. OVERVIEW DOS DADOS
        self._display_data_overview()
        
        # 2. MACHINE LEARNING
        self._display_ml_results()
        
        # 3. CLUSTERING (NOVO - DETALHADO)
        self._display_clustering_results()
        
        # 4. REGRAS DE ASSOCIAÇÃO
        self._display_association_results()
        
        # 5. PERFORMANCE
        self._display_performance_metrics()
        
        print("\n" + "🎯" * 80)
        print("✅ RELATÓRIO COMPLETO EXIBIDO COM SUCESSO!")
        print("🎯" * 80)

    def _display_data_overview(self):
        """Exibir overview dos dados"""
        print(f"\n📊 1. OVERVIEW DOS DADOS")
        print("-" * 50)
        
        if self.df is not None and not self.df.empty:
            if self.is_optimized_run:
                print(f"   📋 Dataset original: ~{self.original_size:,} registros")
                print(f"   🎯 Amostra analisada: {len(self.df):,} registros")
                print(f"   ⚡ Modo: Otimizado para performance")
            else:
                print(f"   📋 Total de registros: {len(self.df):,}")
            
            print(f"   📊 Colunas: {len(self.df.columns)}")
            
            # Distribuição da variável target
            if 'salary' in self.df.columns:
                salary_dist = self.df['salary'].value_counts()
                print(f"   💰 Distribuição salarial:")
                for value, count in salary_dist.items():
                    pct = (count / len(self.df)) * 100
                    bar = "█" * int(pct / 5)
                    print(f"      {value}: {count:,} ({pct:.1f}%) {bar}")
        else:
            print("   ❌ Nenhum dado carregado")

    def _display_ml_results(self):
        """Exibir resultados de ML"""
        print(f"\n🤖 2. MACHINE LEARNING")
        print("-" * 50)
        
        ml_results = self.results.get('ml', {})
        models_results = ml_results.get('models', {})
        
        if models_results:
            print(f"✅ Modelos treinados: {len(models_results)}")
            
            # Tabela de resultados
            print(f"\n{'Modelo':<20} {'Accuracy':<10} {'Status':<15} {'Visualização':<20}")
            print("-" * 70)
            
            best_model = None
            best_score = 0
            
            for model_name, metrics in models_results.items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    accuracy = metrics['accuracy']
                    
                    # Status baseado na accuracy
                    if accuracy > 0.90:
                        status = "🏆 EXCELENTE"
                    elif accuracy > 0.85:
                        status = "✅ MUITO BOM"
                    elif accuracy > 0.80:
                        status = "⚠️ BOM"
                    else:
                        status = "❌ REGULAR"
                    
                    # Barra visual
                    bar_length = int(accuracy * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    
                    print(f"{model_name:<20} {accuracy:<10.3f} {status:<15} {bar}")
                    
                    if accuracy > best_score:
                        best_score = accuracy
                        best_model = model_name
            
            if best_model:
                print(f"\n🏆 MELHOR MODELO: {best_model} ({best_score:.3f} accuracy)")
                
                # Interpretação
                if best_score > 0.85:
                    print("💡 EXCELENTE desempenho - modelo confiável para produção")
                elif best_score > 0.80:
                    print("💡 BOM desempenho - modelo adequado com monitoramento")
                else:
                    print("💡 Desempenho REGULAR - considere melhorias nos dados ou modelo")
        else:
            print("❌ Nenhum modelo foi treinado com sucesso")

    def _display_clustering_results(self):
        """Exibir resultados detalhados de clustering"""
        print(f"\n🎯 3. ANÁLISE DE CLUSTERING")
        print("-" * 50)
        
        clustering_results = self.results.get('clustering', {})
        
        if clustering_results:
            print(f"✅ Métodos executados: {len(clustering_results)}")
            
            # Comparação entre métodos
            print(f"\n📊 COMPARAÇÃO ENTRE MÉTODOS:")
            print(f"{'Método':<15} {'Clusters':<10} {'Silhouette':<12} {'Ruído':<10} {'Status':<15} {'Visualização'}")
            print("-" * 85)
            
            best_method = None
            best_score = -1
            
            for method, result in clustering_results.items():
                if result and isinstance(result, dict):
                    clusters = result.get('n_clusters', 0)
                    silhouette = result.get('silhouette_score', 0)
                    noise_pct = result.get('noise_percentage', 0)
                    has_noise = result.get('has_noise', False)
                    
                    # Status baseado no silhouette score
                    if silhouette > 0.7:
                        status = "🏆 EXCELENTE"
                    elif silhouette > 0.5:
                        status = "✅ BOM"
                    elif silhouette > 0.3:
                        status = "⚠️ REGULAR"
                    else:
                        status = "❌ FRACO"
                    
                    # Barra visual para silhouette score
                    bar_length = int(silhouette * 20) if silhouette > 0 else 0
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    
                    noise_str = f"{noise_pct:.1f}%" if has_noise else "N/A"
                    
                    print(f"{method:<15} {clusters:<10} {silhouette:<12.3f} {noise_str:<10} {status:<15} {bar}")
                    
                    if silhouette > best_score:
                        best_score = silhouette
                        best_method = method
            
            # Melhor método
            if best_method and best_score > 0:
                print(f"\n🏆 MELHOR MÉTODO: {best_method}")
                print(f"🎯 SILHOUETTE SCORE: {best_score:.3f}")
                
                best_result = clustering_results[best_method]
                print(f"📊 CLUSTERS IDENTIFICADOS: {best_result.get('n_clusters', 0)}")
                
                if best_result.get('has_noise', False):
                    noise_pct = best_result.get('noise_percentage', 0)
                    print(f"🔍 OUTLIERS DETECTADOS: {noise_pct:.1f}% dos dados")
                    
                    if noise_pct > 10:
                        print("💡 Alto percentual de outliers - dados heterogêneos")
                    elif noise_pct > 5:
                        print("💡 Outliers moderados - padrão esperado")
                    else:
                        print("💡 Poucos outliers - dados bem estruturados")
                
                # Interpretação dos clusters
                print(f"\n💡 INTERPRETAÇÃO:")
                n_clusters = best_result.get('n_clusters', 0)
                if n_clusters == 2:
                    print("   • 2 grupos: Segmentação binária (ex: salário alto/baixo)")
                elif n_clusters == 3:
                    print("   • 3 grupos: Baixo, médio, alto (segmentação clássica)")
                elif n_clusters >= 4:
                    print(f"   • {n_clusters} grupos: Segmentação detalhada")
                
                if method == "DBSCAN":
                    print("   • DBSCAN detecta grupos por densidade")
                    print("   • Excelente para identificar outliers")
                elif method == "K-Means":
                    print("   • K-Means cria grupos balanceados")
                    print("   • Ideal para segmentação de mercado")
            
        else:
            print("❌ Nenhuma análise de clustering foi executada")
            print("💡 Execute com clustering_pipeline disponível")

    def _display_association_results(self):
        """Exibir resultados de regras de associação"""
        print(f"\n📋 4. REGRAS DE ASSOCIAÇÃO")
        print("-" * 50)
        
        association_results = self.results.get('association_rules', {})
        
        if association_results:
            print(f"✅ Algoritmos executados: Apriori, FP-Growth, Eclat")
            
            # Comparação entre algoritmos
            print(f"\n📊 COMPARAÇÃO ENTRE ALGORITMOS:")
            print(f"{'Algoritmo':<12} {'Regras':<8} {'Conf.Média':<12} {'Lift Médio':<12} {'Status':<15}")
            print("-" * 65)
            
            best_algorithm = None
            best_rules_count = 0
            
            for alg_name in ['apriori', 'fp_growth', 'eclat']:
                if alg_name in association_results and association_results[alg_name].get('rules'):
                    alg_data = association_results[alg_name]
                    rules = alg_data['rules']
                    rules_count = len(rules)
                    
                    if rules_count > 0:
                        avg_confidence = np.mean([r.get('confidence', 0) for r in rules])
                        avg_lift = np.mean([r.get('lift', 0) for r in rules])
                        status = "✅ SUCESSO"
                        
                        # Barra visual para número de regras
                        bar_length = int(rules_count / 10) if rules_count > 0 else 0
                        bar = "█" * min(bar_length, 20)
                        
                        print(f"{alg_name.upper():<12} {rules_count:<8} {avg_confidence:<12.3f} {avg_lift:<12.3f} {status:<15}")
                        print(f"{'':>25} {bar}")
                        
                        if rules_count > best_rules_count:
                            best_rules_count = rules_count
                            best_algorithm = alg_name.upper()
                    else:
                        print(f"{alg_name.upper():<12} {0:<8} {0:<12.3f} {0:<12.3f} {'❌ SEM REGRAS':<15}")
                else:
                    print(f"{alg_name.upper():<12} {0:<8} {0:<12.3f} {0:<12.3f} {'❌ FALHOU':<15}")
            
            # Melhor algoritmo
            if best_algorithm and best_rules_count > 0:
                print(f"\n🏆 MELHOR ALGORITMO: {best_algorithm}")
                print(f"🎯 REGRAS ENCONTRADAS: {best_rules_count}")
                
                # Mostrar top 5 regras do melhor algoritmo
                best_alg_data = association_results.get(best_algorithm.lower(), {})
                if best_alg_data.get('rules'):
                    rules = best_alg_data['rules']
                    print(f"\n🔝 TOP 5 REGRAS ({best_algorithm}):")
                    print(f"{'#':<3} {'Confiança':<12} {'Lift':<8} {'Suporte':<10} {'Qualidade':<12}")
                    print("-" * 50)
                    
                    for i, rule in enumerate(rules[:5], 1):
                        confidence = rule.get('confidence', 0)
                        lift = rule.get('lift', 0)
                        support = rule.get('support', 0)
                        
                        # Classificar qualidade da regra
                        if confidence > 0.8 and lift > 1.5:
                            quality = "🏆 EXCELENTE"
                        elif confidence > 0.6 and lift > 1.2:
                            quality = "✅ BOA"
                        else:
                            quality = "⚠️ REGULAR"
                        
                        print(f"{i:<3} {confidence:<12.3f} {lift:<8.2f} {support:<10.4f} {quality:<12}")
            
            # Estatísticas gerais
            print(f"\n📊 ESTATÍSTICAS GERAIS:")
            total_rules = sum(
                len(association_results[alg].get('rules', []))
                for alg in ['apriori', 'fp_growth', 'eclat']
                if alg in association_results
            )
            print(f"   • Total de regras: {total_rules}")
            print(f"   • Algoritmos bem-sucedidos: {sum(1 for alg in ['apriori', 'fp_growth', 'eclat'] if association_results.get(alg, {}).get('rules'))}")
            
            # Interpretação
            print(f"\n💡 INTERPRETAÇÃO:")
            if total_rules > 100:
                print(f"   ✅ {total_rules} regras - Dataset rico em padrões")
                print(f"   📈 Muitas associações descobertas entre variáveis")
            elif total_rules > 20:
                print(f"   ⚠️ {total_rules} regras - Quantidade moderada")
                print(f"   📊 Padrões identificados mas limitados")
            else:
                print(f"   ❌ {total_rules} regras - Poucos padrões")
                print(f"   💡 Considere ajustar parâmetros (suporte/confiança)")
            
            # Arquivos gerados
            print(f"\n📁 ARQUIVOS GERADOS:")
            rule_files = [
                "output/analysis/apriori_rules.csv",
                "output/analysis/fp_growth_rules.csv", 
                "output/analysis/eclat_rules.csv",
                "output/analysis/association_algorithms_comparison.csv"
            ]
            
            for file_path in rule_files:
                from pathlib import Path
                if Path(file_path).exists():
                    size = Path(file_path).stat().st_size / 1024
                    print(f"   ✅ {file_path} ({size:.1f} KB)")
                else:
                    print(f"   ❌ {file_path}")
        
        else:
            print("❌ Nenhuma análise de regras de associação foi executada")
            print("💡 Execute com association_pipeline disponível para ver resultados")

# Função principal
def main():
    """Executar pipeline principal"""
    try:
        pipeline = HybridPipelineSQL(
            force_csv=True,
            log_level="INFO",
            show_results=True,
            auto_optimize=True  # Otimização automática ativada
        )
        
        results = pipeline.run()
        
        if results:
            print("\n✅ PIPELINE EXECUTADO COM SUCESSO!")
        else:
            print("\n❌ ERRO NA EXECUÇÃO DO PIPELINE")
            
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()