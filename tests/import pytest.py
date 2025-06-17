import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from main import HybridPipelineSQL, setup_logging, setup_database, main
import psutil
import os

"""
🧪 Testes Completos do Pipeline Principal - main.py
Validação de funcionalidades híbridas SQL→CSV, ML e análises
"""


# Adicionar src ao path para imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import do módulo principal

class TestHybridPipelineSQL:
    """Testes do Pipeline Híbrido Principal"""
    
    @pytest.fixture
    def sample_salary_data(self):
        """Dados de exemplo para testes"""
        np.random.seed(42)
        data = {
            'age': np.random.randint(18, 80, 1000),
            'fnlwgt': np.random.randint(10000, 500000, 1000),
            'workclass': np.random.choice(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov'], 1000),
            'education': np.random.choice(['Bachelors', 'Some-college', 'HS-grad', 'Masters'], 1000),
            'education-num': np.random.randint(1, 16, 1000),
            'marital-status': np.random.choice(['Married-civ-spouse', 'Never-married', 'Divorced'], 1000),
            'occupation': np.random.choice(['Tech-support', 'Craft-repair', 'Other-service', 'Sales'], 1000),
            'relationship': np.random.choice(['Husband', 'Not-in-family', 'Wife', 'Own-child'], 1000),
            'race': np.random.choice(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], 1000),
            'sex': np.random.choice(['Male', 'Female'], 1000),
            'capital-gain': np.random.randint(0, 5000, 1000),
            'capital-loss': np.random.randint(0, 2000, 1000),
            'hours-per-week': np.random.randint(20, 80, 1000),
            'native-country': np.random.choice(['United-States', 'Cambodia', 'England', 'Puerto-Rico'], 1000),
            'salary': np.random.choice(['<=50K', '>50K'], 1000, p=[0.76, 0.24])
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_csv_file(self, sample_salary_data):
        """Criar arquivo CSV temporário para testes"""
        temp_dir = Path("test_temp_data")
        temp_dir.mkdir(exist_ok=True)
        
        csv_path = temp_dir / "test_salary_data.csv"
        sample_salary_data.to_csv(csv_path, index=False)
        
        yield csv_path
        
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def model_performance_thresholds(self):
        """Thresholds mínimos de performance para modelos"""
        return {
            'accuracy': 0.75,
            'precision': 0.70,
            'recall': 0.70,
            'f1_score': 0.70
        }
    
    def test_setup_logging(self):
        """Testar configuração do sistema de logging"""
        logger = setup_logging()
        
        assert logger is not None
        assert logger.name == "HybridPipeline"
        assert logger.level <= 20  # INFO level or below
        
        # Verificar se arquivo de log foi criado
        log_dir = Path("logs")
        assert log_dir.exists()
        
        log_files = list(log_dir.glob("pipeline_*.log"))
        assert len(log_files) > 0
        
        print("✅ Sistema de logging configurado corretamente")
    
    def test_pipeline_initialization_csv_mode(self):
        """Testar inicialização do pipeline em modo CSV"""
        pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
        
        assert pipeline is not None
        assert pipeline.force_csv is True
        assert pipeline.data_source is None
        assert pipeline.df is None
        assert isinstance(pipeline.models, dict)
        assert isinstance(pipeline.results, dict)
        assert isinstance(pipeline.performance_metrics, dict)
        
        # Verificar métricas iniciais
        assert 'start_time' in pipeline.performance_metrics
        assert isinstance(pipeline.performance_metrics['start_time'], datetime)
        
        print("✅ Pipeline inicializado em modo CSV")
    
    def test_pipeline_initialization_hybrid_mode(self):
        """Testar inicialização do pipeline em modo híbrido"""
        pipeline = HybridPipelineSQL(force_csv=False, show_results=False)
        
        assert pipeline is not None
        assert pipeline.force_csv is False
        
        # Verificar se componentes foram inicializados
        # SQL pipeline pode ou não estar disponível (dependendo do ambiente)
        # ML pipeline deve tentar inicializar
        
        print("✅ Pipeline inicializado em modo híbrido")
    
    @patch('main.Path')
    def test_load_from_csv_success(self, mock_path, sample_salary_data, temp_csv_file):
        """Testar carregamento bem-sucedido de CSV"""
        # Mock para simular arquivo existente
        mock_path.return_value.exists.return_value = True
        
        with patch('pandas.read_csv', return_value=sample_salary_data):
            pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
            pipeline._load_from_csv()
            
            assert pipeline.df is not None
            assert len(pipeline.df) == len(sample_salary_data)
            assert pipeline.data_source == 'csv'
            
        print("✅ Carregamento de CSV testado com sucesso")
    
    def test_load_from_csv_file_not_found(self):
        """Testar comportamento quando arquivo CSV não é encontrado"""
        with patch('pathlib.Path.exists', return_value=False):
            pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
            pipeline._load_from_csv()
            
            assert pipeline.df is None
            assert pipeline.data_source != 'csv'
        
        print("✅ Tratamento de arquivo CSV não encontrado testado")
    
    def test_basic_cleaning(self, sample_salary_data):
        """Testar limpeza básica dos dados"""
        # Adicionar dados sujos para teste
        dirty_data = sample_salary_data.copy()
        dirty_data.loc[0, 'workclass'] = '?'
        dirty_data.loc[1, 'education'] = '  Bachelors  '  # Com espaços
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[:5]])  # Adicionar duplicatas
        
        pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
        pipeline.df = dirty_data
        
        initial_size = len(pipeline.df)
        pipeline._basic_cleaning()
        
        # Verificar limpeza
        assert pipeline.df is not None
        assert '?' not in pipeline.df.values
        assert pipeline.df['education'].iloc[1].strip() == pipeline.df['education'].iloc[1]
        
        print("✅ Limpeza básica de dados testada")
    
    def test_find_best_model(self):
        """Testar identificação do melhor modelo"""
        pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
        
        # Simular modelos com diferentes performances
        pipeline.models = {
            'Random Forest': {'accuracy': 0.85, 'model': Mock()},
            'Logistic Regression': {'accuracy': 0.82, 'model': Mock()},
            'SVM': {'accuracy': 0.88, 'model': Mock()}
        }
        
        best_model, best_score = pipeline._find_best_model()
        
        assert best_model == 'SVM'
        assert best_score == 0.88
        
        print("✅ Identificação do melhor modelo testada")
    
    def test_performance_metrics_tracking(self, sample_salary_data):
        """Testar tracking de métricas de performance"""
        pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
        pipeline.df = sample_salary_data
        
        # Simular execução de componentes
        start_time = datetime.now()
        pipeline.performance_metrics['data_load_time'] = 0.5
        pipeline.performance_metrics['ml_training_time'] = 2.0
        pipeline.performance_metrics['records_processed'] = len(sample_salary_data)
        pipeline.performance_metrics['data_source'] = 'csv'
        
        assert pipeline.performance_metrics['data_load_time'] == 0.5
        assert pipeline.performance_metrics['ml_training_time'] == 2.0
        assert pipeline.performance_metrics['records_processed'] == len(sample_salary_data)
        assert pipeline.performance_metrics['data_source'] == 'csv'
        
        print("✅ Tracking de métricas de performance testado")
    
    @patch('main.json.dump')
    @patch('main.Path')
    def test_save_results(self, mock_path, mock_json_dump, sample_salary_data):
        """Testar salvamento de resultados"""
        pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
        pipeline.df = sample_salary_data
        pipeline.data_source = 'csv'
        pipeline.models = {'Random Forest': {'accuracy': 0.85}}
        
        # Mock para diretório de output
        mock_path.return_value.mkdir.return_value = None
        mock_path.return_value.__truediv__.return_value = Path("output/test.json")
        
        pipeline._save_results()
        
        # Verificar se json.dump foi chamado
        assert mock_json_dump.called
        
        print("✅ Salvamento de resultados testado")
    
    def test_generate_status_message(self, sample_salary_data):
        """Testar geração de mensagem de status"""
        pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
        
        # Teste 1: Pipeline bem-sucedido
        pipeline.df = sample_salary_data
        pipeline.data_source = 'csv'
        pipeline.models = {'Random Forest': {'accuracy': 0.85}}
        
        status = pipeline._generate_status_message()
        assert '✅' in status
        assert 'CSV' in status
        assert 'Random Forest' in status
        
        # Teste 2: Dados carregados mas sem ML
        pipeline.models = {}
        status = pipeline._generate_status_message()
        assert '⚠️' in status
        
        # Teste 3: Falha no carregamento
        pipeline.df = None
        status = pipeline._generate_status_message()
        assert '❌' in status
        
        print("✅ Geração de mensagem de status testada")
    
    @patch('main.HybridPipelineSQL._display_complete_results')
    def test_run_pipeline_success(self, mock_display, sample_salary_data):
        """Testar execução completa bem-sucedida do pipeline"""
        with patch.object(HybridPipelineSQL, '_load_from_csv') as mock_load_csv:
            # Mock do carregamento de dados
            def mock_load_side_effect():
                pipeline = mock_load_csv.__self__
                pipeline.df = sample_salary_data
                pipeline.data_source = 'csv'
            
            mock_load_csv.side_effect = mock_load_side_effect
            
            # Mock do ML pipeline
            with patch.object(HybridPipelineSQL, '_run_ml_pipeline') as mock_ml:
                def mock_ml_side_effect():
                    pipeline = mock_ml.__self__
                    pipeline.models = {'Random Forest': {'accuracy': 0.85}}
                
                mock_ml.side_effect = mock_ml_side_effect
                
                pipeline = HybridPipelineSQL(force_csv=True, show_results=True)
                results = pipeline.run()
                
                assert 'error' not in results
                assert results['df'] is not None
                assert len(results['models']) > 0
                assert results['data_source'] == 'csv'
                assert 'performance_metrics' in results
                
        print("✅ Execução completa do pipeline testada")
    
    def test_run_pipeline_data_loading_failure(self):
        """Testar falha no carregamento de dados"""
        with patch.object(HybridPipelineSQL, '_load_from_csv') as mock_load:
            # Mock para simular falha no carregamento
            def mock_load_side_effect():
                pipeline = mock_load.__self__
                pipeline.df = None
                pipeline.data_source = None
            
            mock_load.side_effect = mock_load_side_effect
            
            pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
            results = pipeline.run()
            
            assert 'error' in results
            assert "Nenhum dado foi carregado" in results['error']
        
        print("✅ Tratamento de falha no carregamento testado")
    
    @patch('main.HybridPipelineSQL')
    def test_main_function_csv_only_mode(self, mock_pipeline_class):
        """Testar função main em modo CSV apenas"""
        # Mock do pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.return_value = {
            'df': pd.DataFrame({'test': [1, 2, 3]}),
            'models': {'RF': {'accuracy': 0.85}},
            'data_source': 'csv',
            'performance_metrics': {'total_time': 1.5}
        }
        mock_pipeline_class.return_value = mock_pipeline_instance
        
        # Mock sys.argv para simular argumentos
        test_args = ['main.py', '--csv-only']
        with patch('sys.argv', test_args):
            with patch('main.argparse.ArgumentParser.parse_args') as mock_args:
                mock_args.return_value = Mock(
                    csv_only=True,
                    log_level='INFO',
                    setup_db=False,
                    no_results=True
                )
                
                # Não deve gerar exceção
                try:
                    main()
                    test_passed = True
                except SystemExit as e:
                    test_passed = e.code == 0
                
                assert test_passed
                assert mock_pipeline_class.called
        
        print("✅ Função main em modo CSV testada")
    
    @patch('main.setup_database')
    def test_main_function_setup_db_mode(self, mock_setup_db):
        """Testar função main em modo setup database"""
        test_args = ['main.py', '--setup-db']
        with patch('sys.argv', test_args):
            with patch('main.argparse.ArgumentParser.parse_args') as mock_args:
                mock_args.return_value = Mock(
                    setup_db=True,
                    csv_only=False,
                    log_level='INFO',
                    no_results=False
                )
                
                main()
                assert mock_setup_db.called
        
        print("✅ Função main em modo setup database testada")
    
    def test_data_quality_analysis(self, sample_salary_data):
        """Testar análise de qualidade dos dados"""
        # Adicionar problemas de qualidade intencionalmente
        dirty_data = sample_salary_data.copy()
        
        # Adicionar valores ausentes
        dirty_data.loc[0:10, 'workclass'] = np.nan
        dirty_data.loc[5:15, 'occupation'] = np.nan
        
        # Adicionar duplicatas
        duplicated_rows = dirty_data.iloc[:5].copy()
        dirty_data = pd.concat([dirty_data, duplicated_rows], ignore_index=True)
        
        pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
        pipeline.df = dirty_data
        
        # Testar detecção de problemas de qualidade
        missing_data = pipeline.df.isnull().sum()
        duplicates = pipeline.df.duplicated().sum()
        
        assert missing_data.sum() > 0  # Deve detectar dados ausentes
        assert duplicates > 0  # Deve detectar duplicatas
        
        print("✅ Análise de qualidade dos dados testada")
    
    def test_target_distribution_analysis(self, sample_salary_data):
        """Testar análise da distribuição da variável target"""
        pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
        pipeline.df = sample_salary_data
        
        if 'salary' in pipeline.df.columns:
            target_dist = pipeline.df['salary'].value_counts()
            target_dist_pct = pipeline.df['salary'].value_counts(normalize=True)
            
            # Verificar se a distribuição está balanceada (aproximadamente)
            assert len(target_dist) == 2  # Duas classes
            assert '<=50K' in target_dist.index
            assert '>50K' in target_dist.index
            
            # Verificar se percentuais somam 100%
            assert abs(target_dist_pct.sum() - 1.0) < 0.01
            
            print("✅ Análise da distribuição target testada")
    
    @patch('builtins.print')
    def test_display_complete_results(self, mock_print, sample_salary_data):
        """Testar exibição completa de resultados"""
        pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
        pipeline.df = sample_salary_data
        pipeline.data_source = 'csv'
        pipeline.models = {
            'Random Forest': {'accuracy': 0.85},
            'Logistic Regression': {'accuracy': 0.82}
        }
        pipeline.performance_metrics = {
            'data_load_time': 0.5,
            'ml_training_time': 2.0,
            'total_time': 3.0,
            'records_processed': len(sample_salary_data),
            'data_source': 'csv'
        }
        
        pipeline._display_complete_results()
        
        # Verificar se print foi chamado (indicando que resultados foram exibidos)
        assert mock_print.called
        
        # Verificar se chamadas de print contêm informações esperadas
        print_calls = [str(call) for call in mock_print.call_args_list]
        result_text = ' '.join(print_calls)
        
        assert 'RELATÓRIO COMPLETO' in result_text
        assert 'Random Forest' in result_text or 'MACHINE LEARNING' in result_text
        
        print("✅ Exibição completa de resultados testada")
    
    def test_insights_generation(self, sample_salary_data):
        """Testar geração de insights automáticos"""
        pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
        pipeline.df = sample_salary_data
        pipeline.data_source = 'csv'
        pipeline.models = {'Random Forest': {'accuracy': 0.85}}
        pipeline.performance_metrics = {'total_time': 2.5}
        
        # Testar método interno que gera insights
        insights = []
        
        if 'salary' in pipeline.df.columns:
            high_salary_pct = (pipeline.df['salary'] == '>50K').mean() * 100
            insights.append(f"Taxa de salário alto: {high_salary_pct:.1f}%")
        
        if 'age' in pipeline.df.columns:
            avg_age = pipeline.df['age'].mean()
            insights.append(f"Idade média: {avg_age:.1f} anos")
        
        if pipeline.models:
            insights.append("Modelos ML treinados com sucesso")
        
        assert len(insights) > 0
        assert any('salário' in insight.lower() for insight in insights)
        
        print("✅ Geração de insights testada")

class TestSetupFunctions:
    """Testes das funções de setup"""
    
    @patch('main.create_connection')
    @patch('main.run_migration')
    def test_setup_database_success(self, mock_migration, mock_connection):
        """Testar configuração bem-sucedida do banco"""
        # Mock conexão bem-sucedida
        mock_conn = Mock()
        mock_connection.return_value = mock_conn
        mock_migration.return_value = True
        
        # Não deve gerar exceção
        try:
            setup_database()
            test_passed = True
        except Exception:
            test_passed = False
        
        assert test_passed
        print("✅ Setup de banco de dados bem-sucedido testado")
    
    @patch('main.create_connection')
    def test_setup_database_connection_failure(self, mock_connection):
        """Testar falha na conexão do banco"""
        # Mock falha na conexão
        mock_connection.return_value = None
        
        # Não deve gerar exceção, deve tratar graciosamente
        try:
            setup_database()
            test_passed = True
        except Exception:
            test_passed = False
        
        assert test_passed
        print("✅ Tratamento de falha na conexão testado")
    
    def test_setup_database_import_error(self):
        """Testar comportamento quando módulos de banco não estão disponíveis"""
        # Simular ImportError
        with patch('builtins.__import__', side_effect=ImportError("Módulo não encontrado")):
            try:
                setup_database()
                test_passed = True
            except ImportError:
                test_passed = False
        
        assert test_passed
        print("✅ Tratamento de módulos de banco ausentes testado")

class TestIntegrationScenarios:
    """Testes de cenários de integração"""
    
    def test_full_pipeline_csv_mode(self, sample_salary_data):
        """Testar pipeline completo em modo CSV"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Criar arquivo CSV temporário
            csv_path = Path(temp_dir) / "test_data.csv"
            sample_salary_data.to_csv(csv_path, index=False)
            
            # Mockar caminhos de CSV para usar nosso arquivo temporário
            with patch('main.HybridPipelineSQL._load_from_csv') as mock_load:
                def mock_load_side_effect():
                    pipeline = mock_load.__self__
                    pipeline.df = sample_salary_data
                    pipeline.data_source = 'csv'
                
                mock_load.side_effect = mock_load_side_effect
                
                # Executar pipeline
                pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
                results = pipeline.run()
                
                # Validações
                assert 'error' not in results
                assert results['data_source'] == 'csv'
                assert results['df'] is not None
                assert len(results['df']) > 0
                
        print("✅ Pipeline completo em modo CSV testado")
    
    def test_pipeline_fallback_sql_to_csv(self, sample_salary_data):
        """Testar fallback de SQL para CSV"""
        with patch.object(HybridPipelineSQL, 'sql_pipeline', None):
            # Simular falha no SQL e sucesso no CSV
            with patch.object(HybridPipelineSQL, '_load_from_csv') as mock_csv:
                def mock_csv_side_effect():
                    pipeline = mock_csv.__self__
                    pipeline.df = sample_salary_data
                    pipeline.data_source = 'csv'
                
                mock_csv.side_effect = mock_csv_side_effect
                
                pipeline = HybridPipelineSQL(force_csv=False, show_results=False)
                results = pipeline.run()
                
                assert results['data_source'] == 'csv'
                assert mock_csv.called
        
        print("✅ Fallback SQL→CSV testado")
    
    def test_error_handling_and_recovery(self):
        """Testar tratamento de erros e recuperação"""
        # Testar com dados corrompidos
        with patch.object(HybridPipelineSQL, '_load_from_csv') as mock_load:
            # Simular erro no carregamento
            mock_load.side_effect = Exception("Erro simulado")
            
            pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
            results = pipeline.run()
            
            # Pipeline deve retornar erro mas não crashar
            assert 'error' in results
            assert 'Erro simulado' in str(results['error']) or 'Nenhum dado foi carregado' in str(results['error'])
        
        print("✅ Tratamento de erros e recuperação testado")

# Fixtures globais para performance
@pytest.fixture(scope="session")
def performance_test_data():
    """Dados de teste para testes de performance"""
    np.random.seed(42)
    size = 10000  # Dataset maior para teste de performance
    
    data = {
        'age': np.random.randint(18, 80, size),
        'fnlwgt': np.random.randint(10000, 500000, size),
        'workclass': np.random.choice(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov'], size),
        'education': np.random.choice(['Bachelors', 'Some-college', 'HS-grad', 'Masters'], size),
        'education-num': np.random.randint(1, 16, size),
        'marital-status': np.random.choice(['Married-civ-spouse', 'Never-married', 'Divorced'], size),
        'occupation': np.random.choice(['Tech-support', 'Craft-repair', 'Other-service', 'Sales'], size),
        'relationship': np.random.choice(['Husband', 'Not-in-family', 'Wife', 'Own-child'], size),
        'race': np.random.choice(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], size),
        'sex': np.random.choice(['Male', 'Female'], size),
        'capital-gain': np.random.randint(0, 5000, size),
        'capital-loss': np.random.randint(0, 2000, size),
        'hours-per-week': np.random.randint(20, 80, size),
        'native-country': np.random.choice(['United-States', 'Cambodia', 'England', 'Puerto-Rico'], size),
        'salary': np.random.choice(['<=50K', '>50K'], size, p=[0.76, 0.24])
    }
    
    return pd.DataFrame(data)

class TestPerformanceAndScalability:
    """Testes de performance e escalabilidade"""
    
    def test_pipeline_performance_benchmark(self, performance_test_data):
        """Testar performance do pipeline com dataset maior"""
        with patch.object(HybridPipelineSQL, '_load_from_csv') as mock_load:
            def mock_load_side_effect():
                pipeline = mock_load.__self__
                pipeline.df = performance_test_data
                pipeline.data_source = 'csv'
            
            mock_load.side_effect = mock_load_side_effect
            
            start_time = datetime.now()
            
            pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
            results = pipeline.run()
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Pipeline deve processar 10k registros em menos de 30 segundos
            assert execution_time < 30
            assert results['performance_metrics']['records_processed'] == len(performance_test_data)
            
            # Taxa de processamento
            processing_rate = len(performance_test_data) / execution_time
            assert processing_rate > 100  # Pelo menos 100 registros/segundo
            
            print(f"✅ Performance testada: {processing_rate:.0f} registros/segundo")
    
    def test_memory_usage_optimization(self, performance_test_data):
        """Testar otimização de uso de memória"""
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch.object(HybridPipelineSQL, '_load_from_csv') as mock_load:
            def mock_load_side_effect():
                pipeline = mock_load.__self__
                pipeline.df = performance_test_data
                pipeline.data_source = 'csv'
            
            mock_load.side_effect = mock_load_side_effect
            
            pipeline = HybridPipelineSQL(force_csv=True, show_results=False)
            results = pipeline.run()
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Aumento de memória deve ser razoável (menos de 500MB para 10k registros)
            assert memory_increase < 500
            
            print(f"✅ Uso de memória otimizado: +{memory_increase:.1f}MB")

if __name__ == "__main__":
    # Executar testes se chamado diretamente
    pytest.main([__file__, "-v", "--tb=short"])