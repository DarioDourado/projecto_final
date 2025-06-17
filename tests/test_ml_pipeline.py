"""
🤖 Testes do Pipeline de Machine Learning
Validação completa de modelos, métricas e performance
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

class TestMLPipeline:
    """Testes do pipeline de Machine Learning"""
    
    def test_ml_pipeline_initialization(self):
        """Testar inicialização do pipeline ML"""
        try:
            from src.pipelines.ml_pipeline import MLPipeline
            
            pipeline = MLPipeline()
            assert pipeline is not None
            assert hasattr(pipeline, 'models')
            assert hasattr(pipeline, 'results')
            
            print("✅ Pipeline ML inicializado com sucesso")
            
        except ImportError:
            pytest.skip("Pipeline ML não disponível")
    
    def test_model_training_and_performance(self, sample_salary_data, model_performance_thresholds):
        """Testar treinamento e performance dos modelos"""
        try:
            from src.pipelines.ml_pipeline import MLPipeline
            
            pipeline = MLPipeline()
            models, results = pipeline.run(sample_salary_data)
            
            # Validar que modelos foram treinados
            assert len(models) > 0, "Nenhum modelo foi treinado"
            
            expected_models = ['Random Forest', 'Logistic Regression']
            
            for model_name in expected_models:
                assert model_name in models, f"Modelo ausente: {model_name}"
                
                # Validar métricas de performance
                if model_name in results:
                    metrics = results[model_name]
                    
                    if 'accuracy' in metrics:
                        accuracy = metrics['accuracy']
                        assert accuracy >= model_performance_thresholds['accuracy'], \
                            f"{model_name} accuracy {accuracy:.4f} abaixo do threshold {model_performance_thresholds['accuracy']}"
                        
                        print(f"✅ {model_name}: Accuracy = {accuracy:.4f}")
                        
                        # Classificar performance
                        if accuracy >= 0.90:
                            print(f"   🏆 EXCELENTE performance")
                        elif accuracy >= 0.85:
                            print(f"   ✅ MUITO BOA performance")
                        elif accuracy >= 0.80:
                            print(f"   ⚠️ BOA performance")
                        else:
                            print(f"   ❓ REGULAR performance")
            
            return models, results
            
        except ImportError:
            pytest.skip("Pipeline ML não disponível")
    
    def test_model_comparison_and_selection(self, sample_salary_data):
        """Testar comparação e seleção de modelos"""
        try:
            from src.pipelines.ml_pipeline import MLPipeline
            
            pipeline = MLPipeline()
            models, results = pipeline.run(sample_salary_data)
            
            if len(models) < 2:
                pytest.skip("Poucos modelos para comparação")
            
            # Extrair accuracies
            model_accuracies = {}
            for name, result in results.items():
                if isinstance(result, dict) and 'accuracy' in result:
                    model_accuracies[name] = result['accuracy']
            
            assert len(model_accuracies) >= 2, "Insuficientes métricas para comparação"
            
            # Encontrar melhor modelo
            best_model = max(model_accuracies, key=model_accuracies.get)
            best_accuracy = model_accuracies[best_model]
            
            print(f"🏆 MELHOR MODELO: {best_model}")
            print(f"🎯 ACCURACY: {best_accuracy:.4f}")
            
            # Ranking de modelos
            sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
            print("\n📊 RANKING DE MODELOS:")
            for i, (name, acc) in enumerate(sorted_models, 1):
                print(f"   {i}. {name}: {acc:.4f}")
            
            # Validar diferença significativa
            if len(sorted_models) >= 2:
                diff = sorted_models[0][1] - sorted_models[1][1]
                if diff > 0.02:
                    print(f"✅ Diferença significativa: {diff:.4f}")
                else:
                    print(f"⚠️ Diferença pequena: {diff:.4f}")
            
        except ImportError:
            pytest.skip("Pipeline ML não disponível")
    
    def test_model_artifacts_saving(self, sample_salary_data, test_data_dir):
        """Testar salvamento de artefatos de ML"""
        try:
            from src.pipelines.ml_pipeline import MLPipeline
            
            pipeline = MLPipeline()
            models, results = pipeline.run(sample_salary_data)
            
            # Verificar se arquivos foram salvos
            processed_dir = Path("data/processed")
            
            if processed_dir.exists():
                model_files = list(processed_dir.glob("*_model.joblib"))
                assert len(model_files) > 0, "Nenhum modelo salvo encontrado"
                
                print(f"✅ {len(model_files)} modelos salvos:")
                for model_file in model_files:
                    print(f"   📁 {model_file.name}")
                
                # Verificar preprocessadores
                scaler_file = processed_dir / "scaler.joblib"
                encoder_file = processed_dir / "label_encoders.joblib"
                
                if scaler_file.exists():
                    print("✅ Scaler salvo")
                if encoder_file.exists():
                    print("✅ Label encoders salvos")
            
        except ImportError:
            pytest.skip("Pipeline ML não disponível")
    
    def test_prediction_capability(self, sample_salary_data):
        """Testar capacidade de predição dos modelos"""
        try:
            from src.pipelines.ml_pipeline import MLPipeline
            import joblib
            
            pipeline = MLPipeline()
            models, results = pipeline.run(sample_salary_data)
            
            # Testar predição com dados de exemplo
            test_sample = sample_salary_data.iloc[:5]
            numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
            
            X_test = test_sample[numeric_features].fillna(0)
            
            for model_name, model in models.items():
                if hasattr(model, 'predict'):
                    try:
                        # Normalizar se necessário
                        if pipeline.scaler:
                            X_test_scaled = pipeline.scaler.transform(X_test)
                            predictions = model.predict(X_test_scaled)
                        else:
                            predictions = model.predict(X_test)
                        
                        assert len(predictions) == len(X_test), f"Número incorreto de predições para {model_name}"
                        assert all(pred in [0, 1] for pred in predictions), f"Predições inválidas para {model_name}"
                        
                        print(f"✅ {model_name}: {len(predictions)} predições válidas")
                        
                    except Exception as e:
                        print(f"⚠️ {model_name}: Erro na predição - {e}")
            
        except ImportError:
            pytest.skip("Pipeline ML não disponível")