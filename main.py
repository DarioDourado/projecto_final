"""
Pipeline Principal Reorganizado - Sistema Modular
Versão limpa e organizada do sistema de análise salarial
"""

import sys
import logging
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logger import setup_logging
from src.visualization.styles import setup_matplotlib_style
from src.pipelines.data_pipeline import DataPipeline
from src.pipelines.ml_pipeline import MLPipeline
from src.pipelines.analysis_pipeline import AnalysisPipeline
from src.pipelines.performance_pipeline import PerformancePipeline
from src.pipelines.utils import check_data_structure

class MasterPipeline:
    """Pipeline principal que coordena todos os outros pipelines"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.data_pipeline = DataPipeline()
        self.ml_pipeline = MLPipeline()
        self.analysis_pipeline = AnalysisPipeline()
        self.performance_pipeline = PerformancePipeline()
        
        # Resultados
        self.df = None
        self.models = {}
        self.results = {}
        self.best_k = None
        self.rules = []
    
    def run(self):
        """Executar pipeline completo"""
        logging.info("🚀 Sistema de Análise Salarial - VERSÃO MODULAR")
        logging.info("="*60)
        
        try:
            # 0. Verificações iniciais
            self._setup()
            
            # 1. Pipeline de dados
            self.df = self.data_pipeline.run()
            
            # 2. Pipeline de ML
            self.models, self.results = self.ml_pipeline.run(self.df)
            
            # 3. Pipeline de performance
            self.performance_pipeline.run(self.models, self.results, self.df)
            
            # 4. Pipelines de análise
            self.best_k = self.analysis_pipeline.run_clustering(self.df)
            self.rules = self.analysis_pipeline.run_association_rules(self.df)
            self.analysis_pipeline.run_advanced_metrics(self.df, self.results)
            
            # 5. Relatório final
            self._generate_final_report()
            
            logging.info("🎉 Pipeline concluído com sucesso!")
            logging.info("📊 Dashboard: streamlit run dashboard_app.py")
            
        except Exception as e:
            logging.error(f"❌ Erro durante execução: {e}")
            raise
    
    def _setup(self):
        """Configurações iniciais"""
        check_data_structure()
        setup_matplotlib_style()
        logging.info("✅ Configurações iniciais concluídas")
    
    def _generate_final_report(self):
        """Gerar relatório final consolidado"""
        logging.info("\n" + "="*60)
        logging.info("📊 RELATÓRIO FINAL")
        logging.info("="*60)
        
        logging.info(f"📊 Dataset: {len(self.df)} registos")
        logging.info("🤖 Modelos treinados:")
        for name, result in self.results.items():
            accuracy = result.get('accuracy', 0)
            logging.info(f"  • {name}: {accuracy:.4f}")
        
        if self.best_k:
            logging.info(f"🎯 Clustering: {self.best_k} clusters")
        
        rules_count = len(self.rules) if hasattr(self.rules, '__len__') else 0
        logging.info(f"📋 Regras de associação: {rules_count}")
        
        # Verificar arquivos gerados
        self._show_generated_files()
    
    def _show_generated_files(self):
        """Mostrar arquivos gerados"""
        output_dir = Path("output")
        
        if output_dir.exists():
            images_dir = output_dir / "images"
            analysis_dir = output_dir / "analysis"
            
            if images_dir.exists():
                image_files = list(images_dir.glob("*.png"))
                logging.info(f"\n🎨 {len(image_files)} visualizações em output/images/")
                for img in image_files[:3]:
                    logging.info(f"  • {img.name}")
                if len(image_files) > 3:
                    logging.info(f"  • ... e mais {len(image_files) - 3}")
            
            if analysis_dir.exists():
                analysis_files = list(analysis_dir.glob("*"))
                logging.info(f"\n📊 {len(analysis_files)} análises em output/analysis/")
                for file in analysis_files:
                    logging.info(f"  • {file.name}")
        
        processed_dir = Path("data/processed")
        if processed_dir.exists():
            joblib_files = list(processed_dir.glob("*.joblib"))
            logging.info(f"\n📁 {len(joblib_files)} modelos em data/processed/")

def main():
    """Função principal"""
    pipeline = MasterPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()