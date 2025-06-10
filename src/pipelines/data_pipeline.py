"""Pipeline de processamento de dados"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataPipeline:
    """Pipeline de processamento de dados"""
    
    def __init__(self):
        self.df = None
    
    def run(self):
        """Executar pipeline de dados"""
        try:
            # Import local para evitar dependências circulares
            from src.data.processor import DataProcessor
            from src.utils.readability_insights import show_readability_insights
            
            logger.info("🔧 Iniciando processamento de dados...")
            
            # Processar dados
            processor = DataProcessor()
            self.df = processor.process_complete_pipeline()
            
            # Mostrar insights de legibilidade
            try:
                show_readability_insights()
            except ImportError:
                logger.info("📊 Insights de legibilidade não disponíveis")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao mostrar insights: {e}")
            
            logger.info(f"✅ Dados processados: {len(self.df)} registos")
            return self.df
            
        except Exception as e:
            logger.error(f"❌ Erro no pipeline de dados: {e}")
            raise