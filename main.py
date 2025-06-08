"""Pipeline principal reorganizado - mantendo funcionalidades do projeto_salario.py"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

# Imports absolutos para evitar problemas
from src.utils.logger import setup_logging
from src.data.processor import DataProcessor
from src.visualization.styles import setup_matplotlib_style
from src.visualization.plots import VisualizationGenerator
from src.models.trainer import ModelTrainer
import logging

def check_data_structure():
    """Verificar se a estrutura de dados está correta"""
    project_root = Path(__file__).parent
    data_file = project_root / "data" / "raw" / "4-Carateristicas_salario.csv"
    
    if not data_file.exists():
        logging.warning("⚠️ Estrutura de dados não configurada corretamente!")
        logging.info("🔧 Execute primeiro: python setup_data_structure.py")
        
        # Tentar configurar automaticamente
        try:
            import setup_data_structure
            setup_data_structure.setup_data_structure()
            if data_file.exists():
                logging.info("✅ Estrutura configurada automaticamente!")
            else:
                raise FileNotFoundError("Não foi possível configurar a estrutura automaticamente")
        except Exception as e:
            logging.error(f"❌ Erro ao configurar estrutura: {e}")
            logging.error("Manual: Coloque '4-Carateristicas_salario.csv' em 'data/raw/'")
            raise

def main():
    """Pipeline principal mantendo todas as funcionalidades"""
    # Configurar logging
    logger = setup_logging()
    logging.info("🚀 Iniciando Sistema de Análise Salarial v2.0")
    logging.info("="*60)
    
    try:
        # 0. Verificar estrutura de dados
        check_data_structure()
        
        # 1. Configurar estilo de visualização
        setup_matplotlib_style()
        logging.info("✅ Estilo de visualização configurado")
        
        # 2. Processar dados (de data/raw/ para data/processed/)
        processor = DataProcessor()
        df = processor.process_complete_pipeline()
        
        # 3. Treinar modelos (salvar em data/processed/)
        trainer = ModelTrainer()
        models, results = trainer.train_complete_pipeline(df)
        
        # 4. Gerar visualizações (incluindo feature importance)
        viz_generator = VisualizationGenerator()
        viz_generator.generate_all_plots(df, models=models, results=results)
        
        # 5. Relatório final
        logging.info("\n" + "="*60)
        logging.info("RELATÓRIO FINAL")
        logging.info("="*60)
        
        logging.info(f"📊 Dataset final: {len(df)} registros")
        logging.info("🤖 Modelos treinados:")
        for name, result in results.items():
            logging.info(f"  • {name}: {result['accuracy']:.4f}")
        
        logging.info("📂 Estrutura de saídas:")
        logging.info("  📈 Visualizações: output/images/")
        logging.info("  📊 Dados processados: data/processed/")
        logging.info("  🤖 Modelos (.joblib): data/processed/")
        logging.info("  📚 Cópias organizadas: models/trained/")
        logging.info("\n🎉 Pipeline concluído com sucesso!")
        
        # Listar arquivos .joblib gerados
        processed_dir = Path("data/processed")
        joblib_files = list(processed_dir.glob("*.joblib"))
        if joblib_files:
            logging.info("\n📁 Arquivos .joblib gerados:")
            for file in joblib_files:
                logging.info(f"  • {file.name}")
        
        # Listar visualizações geradas
        images_dirs = [Path("output/images"), Path("imagens")]
        for img_dir in images_dirs:
            if img_dir.exists():
                image_files = list(img_dir.glob("*.png"))
                if image_files:
                    logging.info(f"\n🎨 Visualizações geradas em {img_dir}:")
                    for img in image_files:
                        logging.info(f"  • {img.name}")
                break
        
    except Exception as e:
        logging.error(f"❌ Erro durante execução: {e}")
        raise

if __name__ == "__main__":
    main()