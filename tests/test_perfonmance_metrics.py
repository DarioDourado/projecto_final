"""
📈 Testes de Performance e Métricas
Validação detalhada de todas as métricas de Data Science
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent / "src"))

class TestPerformanceMetrics:
    """Testes de métricas de performance"""
    
    def test_pipeline_execution_time(self):
        """Testar tempo de execução do pipeline"""
        try:
            from main import HybridPipelineSQL
            
            start_time = datetime.now()
            
            pipeline = HybridPipelineSQL(force_csv=True, log_level="WARNING")
            results = pipeline.run()
            
            execution_time = datetime.now() - start_time
            
            # Validar que executou em tempo razoável (< 30 segundos)
            assert execution_time.total_seconds() < 30, f"Pipeline muito lento: {execution_time.total_seconds():.2f}s"
            
            # Validar métricas de performance
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                
                assert 'total_time' in metrics
                assert 'data_load_time' in metrics
                assert 'ml_training_time' in metrics
                
                print(f"⏱️ MÉTRICAS DE PERFORMANCE:")
                print(f"   📊 Tempo total: {metrics['total_time']:.2f}s")
                print(f"   📋 Carregamento: {metrics['data_load_time']:.2f}s")
                print(f"   🤖 ML Training: {metrics['ml_training_time']:.2f}s")
                print(f"   📈 Registros: {metrics['records_processed']:,}")
            
        except ImportError:
            pytest.skip("Pipeline principal não disponível")
    
    def test_data_science_results_validation(self):
        """Validar todos os resultados importantes de Data Science"""
        try:
            from main import HybridPipelineSQL
            
            pipeline = HybridPipelineSQL(force_csv=True, log_level="WARNING")
            results = pipeline.run()
            
            # Validar estrutura de resultados
            required_keys = ['df', 'models', 'results', 'data_source', 'performance_metrics', 'status']
            
            for key in required_keys:
                assert key in results, f"Chave ausente nos resultados: {key}"
            
            # Validar dados
            df = results['df']
            assert df is not None and len(df) > 0, "Dataset vazio nos resultados"
            
            # Validar modelos
            models = results['models']
            if len(models) > 0:
                print(f"🤖 MODELOS TREINADOS: {len(models)}")
                
                for name, model in models.items():
                    print(f"   ✅ {name}")
                    
                    # Validar que é um modelo válido
                    assert hasattr(model, 'predict') or isinstance(model, dict), f"Modelo inválido: {name}"
            
            # Validar métricas de qualidade dos dados
            self._validate_data_quality(df)
            
            # Validar fonte de dados
            data_source = results['data_source']
            assert data_source in ['sql', 'csv'], f"Fonte de dados inválida: {data_source}"
            
            print(f"✅ VALIDAÇÃO COMPLETA DOS RESULTADOS:")
            print(f"   📊 Fonte: {data_source.upper()}")
            print(f"   📋 Registros: {len(df):,}")
            print(f"   🤖 Modelos: {len(models)}")
            print(f"   📈 Status: {results['status']}")
            
        except ImportError:
            pytest.skip("Pipeline principal não disponível")
    
    def _validate_data_quality(self, df):
        """Validar qualidade dos dados nos resultados"""
        # Métricas básicas
        total_records = len(df)
        total_columns = len(df.columns)
        
        print(f"📊 QUALIDADE DOS DADOS:")
        print(f"   📋 Registros: {total_records:,}")
        print(f"   📊 Colunas: {total_columns}")
        
        # Tipos de dados
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        print(f"   🔢 Numéricas: {len(numeric_cols)}")
        print(f"   📝 Categóricas: {len(categorical_cols)}")
        
        # Dados ausentes
        missing_data = df.isnull().sum()
        cols_with_missing = missing_data[missing_data > 0]
        
        if len(cols_with_missing) > 0:
            print(f"   ⚠️ Colunas com missing:")
            for col, count in cols_with_missing.items():
                percentage = (count / total_records) * 100
                print(f"     • {col}: {count} ({percentage:.1f}%)")
        else:
            print(f"   ✅ Sem dados ausentes")
        
        # Duplicatas
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            dup_percentage = (duplicates / total_records) * 100
            print(f"   ⚠️ Duplicatas: {duplicates} ({dup_percentage:.1f}%)")
        else:
            print(f"   ✅ Sem duplicatas")
        
        # Distribuição target
        if 'salary' in df.columns:
            target_dist = df['salary'].value_counts(normalize=True)
            print(f"   🎯 Distribuição target:")
            for value, percentage in target_dist.items():
                print(f"     • {value}: {percentage:.1f}%")
    
    def test_output_artifacts_validation(self):
        """Validar artefatos de saída gerados"""
        output_dir = Path("output")
        
        if output_dir.exists():
            # Estado do pipeline
            state_file = output_dir / "pipeline_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                required_state_keys = ['data_source', 'models_count', 'performance_metrics', 'timestamp']
                for key in required_state_keys:
                    assert key in state, f"Chave ausente no estado: {key}"
                
                print(f"✅ Estado do pipeline validado:")
                print(f"   📊 Fonte: {state['data_source']}")
                print(f"   🤖 Modelos: {state['models_count']}")
                print(f"   📅 Timestamp: {state['timestamp']}")
            
            # Resumo dos modelos
            models_file = output_dir / "models_summary.json"
            if models_file.exists():
                with open(models_file, 'r') as f:
                    models_summary = json.load(f)
                
                print(f"✅ Resumo de modelos validado: {len(models_summary)} modelos")
        
        # Artefatos de ML
        processed_dir = Path("data/processed")
        if processed_dir.exists():
            model_files = list(processed_dir.glob("*_model.joblib"))
            print(f"✅ Artefatos ML: {len(model_files)} modelos salvos")
    
    def test_comprehensive_data_science_report(self):
        """Gerar relatório completo de Data Science"""
        try:
            from main import HybridPipelineSQL
            
            print("\n" + "="*80)
            print("📊 RELATÓRIO COMPLETO DE DATA SCIENCE")
            print("="*80)
            
            pipeline = HybridPipelineSQL(force_csv=True, log_level="WARNING")
            results = pipeline.run()
            
            # Seção 1: Dados
            print("\n📋 1. DADOS:")
            df = results['df']
            print(f"   • Registros: {len(df):,}")
            print(f"   • Features: {len(df.columns)}")
            print(f"   • Fonte: {results['data_source'].upper()}")
            
            # Seção 2: Modelos
            print("\n🤖 2. MACHINE LEARNING:")
            models = results['models']
            print(f"   • Modelos treinados: {len(models)}")
            
            # Seção 3: Performance
            print("\n📈 3. PERFORMANCE:")
            metrics = results['performance_metrics']
            print(f"   • Tempo total: {metrics['total_time']:.2f}s")
            print(f"   • Registros processados: {metrics['records_processed']:,}")
            
            # Seção 4: Qualidade
            print("\n✅ 4. QUALIDADE:")
            missing_data = df.isnull().sum().sum()
            duplicates = df.duplicated().sum()
            print(f"   • Dados ausentes: {missing_data}")
            print(f"   • Duplicatas: {duplicates}")
            
            print("\n" + "="*80)
            print("✅ RELATÓRIO CONCLUÍDO")
            print("="*80)
            
        except ImportError:
            pytest.skip("Pipeline principal não disponível")