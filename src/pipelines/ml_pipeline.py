"""Pipeline de Machine Learning"""

import logging
from pathlib import Path
from src.models.trainer import ModelTrainer
from src.visualization.plots import VisualizationGenerator

logger = logging.getLogger(__name__)

class MLPipeline:
    """Pipeline de Machine Learning"""
    
    def __init__(self):
        self.trainer = ModelTrainer()
        self.viz_generator = VisualizationGenerator()
        self.models = {}
        self.results = {}
    
    def run(self, df):
        """Executar pipeline de ML"""
        logger.info("🤖 TREINO DE MODELOS")
        logger.info("="*60)
        
        # Treinar modelos
        self.models, self.results = self.trainer.train_complete_pipeline(df)
        
        logger.info("📈 GERAÇÃO DE VISUALIZAÇÕES")
        logger.info("="*60)
        
        # Gerar visualizações
        self.viz_generator.generate_all_plots(df, models=self.models, results=self.results)
        
        return self.models, self.results