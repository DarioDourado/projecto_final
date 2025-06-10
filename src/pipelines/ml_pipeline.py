"""Pipeline de Machine Learning"""

import logging

logger = logging.getLogger(__name__)

class MLPipeline:
    """Pipeline de Machine Learning"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def run(self, df):
        """Executar pipeline de ML"""
        try:
            # Import local para evitar dependências circulares
            from src.models.trainer import ModelTrainer
            from src.visualization.plots import VisualizationGenerator
            
            logger.info("🤖 Iniciando treino de modelos...")
            
            # Treinar modelos
            trainer = ModelTrainer()
            self.models, self.results = trainer.train_complete_pipeline(df)
            
            logger.info("📈 Gerando visualizações...")
            
            # Gerar visualizações
            viz_generator = VisualizationGenerator()
            viz_generator.generate_all_plots(df, models=self.models, results=self.results)
            
            logger.info(f"✅ Pipeline ML concluído: {len(self.models)} modelos treinados")
            return self.models, self.results
            
        except Exception as e:
            logger.error(f"❌ Erro no pipeline de ML: {e}")
            raise