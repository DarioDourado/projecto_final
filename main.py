"""
Pipeline Principal SQL-Only - Sistema Exclusivo de Banco de Dados
Versão sem dependência de CSV
"""

import sys
import logging
import os
from pathlib import Path
import argparse

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

# Imports básicos
from src.utils.logger import setup_logging
from src.visualization.styles import setup_matplotlib_style

class MasterPipelineSQL:
    """Pipeline principal exclusivo para banco de dados"""
    
    def __init__(self, force_migration: bool = False):
        self.logger = setup_logging()
        self.force_migration = force_migration
        
        # Verificar configuração de banco OBRIGATÓRIA
        if not self._check_database_config():
            raise ValueError("❌ Configuração de banco de dados obrigatória!")
        
        # Import dos pipelines SQL com tratamento de erro
        try:
            from src.pipelines.data_pipeline import DataPipelineSQL
            from src.pipelines.ml_pipeline import MLPipeline
            from src.pipelines.analysis_pipeline import AnalysisPipeline
            from src.pipelines.performance_pipeline import PerformancePipeline
            
            self.data_pipeline = DataPipelineSQL(force_migration=force_migration)
            self.ml_pipeline = MLPipeline()
            self.analysis_pipeline = AnalysisPipeline()
            self.performance_pipeline = PerformancePipeline()
        except ImportError as e:
            self.logger.error(f"❌ Erro ao importar pipelines: {e}")
            raise
        
        # Resultados
        self.df = None
        self.models = {}
        self.results = {}
        self.best_k = None
        self.rules = []
    
    def _check_database_config(self) -> bool:
        """Verificar configuração obrigatória do banco"""
        required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logging.error(f"❌ Variáveis de ambiente obrigatórias ausentes: {missing_vars}")
            logging.info("💡 Configure as variáveis de ambiente:")
            logging.info("   export DB_HOST=localhost")
            logging.info("   export DB_NAME=salary_analysis")
            logging.info("   export DB_USER=salary_user")
            logging.info("   export DB_PASSWORD=sua_senha")
            return False
        
        logging.info("✅ Configuração de banco de dados encontrada")
        return True
    
    def run(self):
        """Executar pipeline completo SQL-only com tratamento de erros"""
        logging.info("🚀 Sistema de Análise Salarial - VERSÃO SQL EXCLUSIVA")
        logging.info("🗄️ Fonte de dados: BANCO DE DADOS MySQL")
        logging.info("="*60)
        
        try:
            # 0. Configurações iniciais
            self._setup()
            
            # 1. Pipeline de dados SQL
            logging.info("\n📊 PIPELINE DE DADOS SQL")
            logging.info("-" * 40)
            self.df = self.data_pipeline.run()
            
            if self.df is None or len(self.df) == 0:
                raise ValueError("❌ Nenhum dado carregado do banco")
            
            logging.info(f"✅ Dados carregados: {len(self.df)} registros")
            
            # 2. Pipeline de ML com tratamento de erro
            logging.info("\n🤖 PIPELINE DE ML")
            logging.info("-" * 40)
            
            try:
                self.models, self.results = self.ml_pipeline.run(self.df)
                
                if not self.models:
                    logging.warning("⚠️ Nenhum modelo foi treinado com sucesso")
                else:
                    logging.info(f"✅ {len(self.models)} modelos treinados")
                    
            except Exception as e:
                logging.error(f"❌ Erro no pipeline ML: {e}")
                logging.warning("⚠️ Continuando sem modelos ML...")
                self.models = {}
                self.results = {}
            
            # 3. Pipeline de performance (apenas se houver modelos)
            if self.models:
                logging.info("\n📈 PIPELINE DE PERFORMANCE")
                logging.info("-" * 40)
                try:
                    self.performance_pipeline.run(self.models, self.results, self.df)
                except Exception as e:
                    logging.error(f"❌ Erro no pipeline de performance: {e}")
            
            # 4. Pipelines de análise
            logging.info("\n🎯 PIPELINE DE ANÁLISES")
            logging.info("-" * 40)
            
            try:
                self.best_k = self.analysis_pipeline.run_clustering(self.df)
            except Exception as e:
                logging.error(f"❌ Erro no clustering: {e}")
                self.best_k = None
            
            try:
                self.rules = self.analysis_pipeline.run_association_rules(self.df)
            except Exception as e:
                logging.error(f"❌ Erro nas regras de associação: {e}")
                self.rules = []
            
            try:
                self.analysis_pipeline.run_advanced_metrics(self.df, self.results)
            except Exception as e:
                logging.error(f"❌ Erro nas métricas avançadas: {e}")
            
            # 5. Criar views SQL de análise
            logging.info("\n🗄️ CRIANDO VIEWS DE ANÁLISE")
            logging.info("-" * 40)
            try:
                self.data_pipeline.create_analysis_views()
            except Exception as e:
                logging.error(f"❌ Erro ao criar views SQL: {e}")
            
            # 6. Relatório final
            self._generate_final_report()
            
            logging.info("🎉 Pipeline SQL concluído!")
            logging.info("📊 Dashboard: streamlit run app.py")
            
        except Exception as e:
            logging.error(f"❌ Erro crítico durante execução: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
            # Tentar relatório de emergência
            try:
                self._emergency_report()
            except:
                pass
            
            raise
    
    def _setup(self):
        """Configurações iniciais SQL-only"""
        try:
            setup_matplotlib_style()
            
            # Testar conexão com banco
            self._test_database_connection()
            
            logging.info("✅ Configurações SQL iniciais concluídas")
        except Exception as e:
            logging.error(f"❌ Erro na configuração: {e}")
            raise
    
    def _test_database_connection(self):
        """Testar conexão com banco de dados"""
        try:
            from src.database.connection import DatabaseConnection
            
            with DatabaseConnection() as db:
                result = db.execute_query("SELECT 1 as test")
                if result:
                    logging.info("✅ Conexão com banco de dados funcionando")
                else:
                    raise ValueError("Conexão falhou")
                    
        except Exception as e:
            logging.error(f"❌ Erro na conexão com banco: {e}")
            raise
    
    def _generate_final_report(self):
        """Gerar relatório final SQL"""
        logging.info("\n" + "="*60)
        logging.info("📊 RELATÓRIO FINAL SQL")
        logging.info("="*60)
        
        # Estatísticas do banco
        try:
            stats = self.data_pipeline.get_statistics_from_sql()
            total_records = stats.get('total_records', 0)
            logging.info(f"📋 Registros no banco: {total_records:,}")
            
            if 'salary_stats' in stats:
                for salary_stat in stats['salary_stats']:
                    salary_range = salary_stat['salary_range']
                    count = salary_stat['count']
                    avg_age = salary_stat['avg_age']
                    logging.info(f"  💰 {salary_range}: {count:,} pessoas (idade média: {avg_age:.1f})")
        
        except Exception as e:
            logging.warning(f"⚠️ Erro nas estatísticas SQL: {e}")
        
        # Dados processados
        if self.df is not None:
            logging.info(f"📊 Dataset ML: {len(self.df)} registros processados")
        
        # Modelos treinados
        logging.info("🤖 Modelos treinados:")
        for name, result in self.results.items():
            accuracy = result.get('accuracy', 0)
            logging.info(f"  • {name}: {accuracy:.4f}")
        
        # Clustering
        if self.best_k:
            logging.info(f"🎯 Clustering: {self.best_k} clusters identificados")
        
        # Regras de associação
        rules_count = len(self.rules) if hasattr(self.rules, '__len__') else 0
        logging.info(f"📋 Regras de associação: {rules_count}")
        
        # Arquivos gerados
        self._show_generated_files()
        
        # Instruções SQL
        logging.info("\n💡 Para análises SQL diretas:")
        logging.info("   • Use as views criadas: high_earners_view, education_analysis_view")
        logging.info("   • Execute queries customizadas via dashboard")
        logging.info("   • Acesse análises em tempo real via SQL")
    
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

    def _emergency_report(self):
        """Relatório de emergência quando pipeline falha"""
        logging.info("\n🚨 RELATÓRIO DE EMERGÊNCIA")
        logging.info("="*40)
        
        if self.df is not None:
            logging.info(f"📊 Dados carregados: {len(self.df)} registros")
            logging.info(f"📋 Colunas: {list(self.df.columns)}")
        else:
            logging.info("❌ Nenhum dado foi carregado")
        
        logging.info(f"🤖 Modelos criados: {len(self.models)}")
        logging.info(f"📊 Resultados: {len(self.results)}")
        
        logging.info("\n💡 Para diagnóstico:")
        logging.info("   • Verificar logs detalhados")
        logging.info("   • Testar conexão: python setup_sql_only.py")
        logging.info("   • Verificar dados: SELECT COUNT(*) FROM person;")

def main():
    """Função principal SQL-only"""
    parser = argparse.ArgumentParser(description="Sistema de Análise Salarial SQL-Only")
    parser.add_argument('--migrate', action='store_true', 
                       help='Forçar migração CSV→SQL (se CSV disponível)')
    parser.add_argument('--setup-db', action='store_true',
                       help='Apenas configurar estrutura do banco')
    
    args = parser.parse_args()
    
    # Verificar variáveis de ambiente
    if not all(os.getenv(var) for var in ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']):
        print("❌ ERRO: Configuração de banco de dados obrigatória!")
        print("\n💡 Configure as variáveis de ambiente:")
        print("export DB_HOST=localhost")
        print("export DB_NAME=salary_analysis")
        print("export DB_USER=salary_user")
        print("export DB_PASSWORD=sua_senha")
        print("\n🔧 Ou crie arquivo .env com essas variáveis")
        return
    
    # Setup apenas da estrutura
    if args.setup_db:
        try:
            from src.database.migration import DatabaseMigrator
            migrator = DatabaseMigrator()
            if migrator.create_database_structure():
                print("✅ Estrutura do banco criada com sucesso!")
            else:
                print("❌ Erro ao criar estrutura do banco")
        except Exception as e:
            print(f"❌ Erro: {e}")
        return
    
    try:
        pipeline = MasterPipelineSQL(force_migration=args.migrate)
        pipeline.run()
    except Exception as e:
        logging.error(f"❌ Erro crítico: {e}")
        print(f"\n❌ ERRO: {e}")
        print("\n💡 Soluções possíveis:")
        print("  1. Verificar conexão com banco: mysql -u salary_user -p salary_analysis")
        print("  2. Criar estrutura: python main.py --setup-db")
        print("  3. Migrar dados: python main.py --migrate")
        print("  4. Verificar variáveis de ambiente")

if __name__ == "__main__":
    main()