"""Pipeline de processamento de dados"""

import logging
from pathlib import Path
from src.utils.logger import setup_logging
from src.data.processor import DataProcessor
from src.utils.readability_insights import show_readability_insights

logger = logging.getLogger(__name__)

class DataPipeline:
    """Pipeline de processamento de dados"""
    
    def __init__(self):
        self.processor = DataProcessor()
        self.df = None
    
    def run(self):
        """Executar pipeline de dados"""
        logger.info("🔧 PROCESSAMENTO DE DADOS")
        logger.info("="*60)
        
        # Processar dados
        self.df = self.processor.process_complete_pipeline()
        
        # Mostrar insights de legibilidade
        try:
            show_readability_insights()
        except ImportError:
            logger.info("📊 Insights de legibilidade não disponíveis")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao mostrar insights: {e}")
        
        return self.df