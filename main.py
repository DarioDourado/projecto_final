"""
Pipeline Principal SQL-Only - Sistema Exclusivo de Banco de Dados
Versão sem dependência de CSV - CORRIGIDA
"""

import sys
import logging
import os
from pathlib import Path
import argparse

# Carregar variáveis de ambiente PRIMEIRO
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Se python-dotenv não estiver instalado, carregar manualmente
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

# Imports básicos
from src.utils.logger import setup_logging
from src.visualization.styles import setup_matplotlib_style

# Imports com fallback para MySQL
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    try:
        import pymysql
        pymysql.install_as_MySQLdb()
        MYSQL_AVAILABLE = True
        print("⚠️  Usando PyMySQL como fallback")
    except ImportError:
        MYSQL_AVAILABLE = False
        print("❌ Nenhum driver MySQL disponível")

# Verificar antes de inicializar pipeline
if not MYSQL_AVAILABLE:
    print("❌ Driver MySQL não encontrado")
    print("💡 Instale: pip install mysql-connector-python")
    sys.exit(1)

class MasterPipelineSQL:
    """Pipeline principal exclusivo para banco de dados"""
    
    def __init__(self, force_migration: bool = False):
        # Configurar logging PRIMEIRO
        self.logger = setup_logging()
        self.force_migration = force_migration
        
        # Verificar configuração de banco OBRIGATÓRIA
        if not self._check_database_config():
            raise ValueError("❌ Configuração de banco de dados obrigatória!")
        
        # Inicializar pipelines com tratamento de erro
        self._initialize_pipelines()
        
        # Resultados
        self.df = None
        self.models = {}
        self.results = {}
        self.best_k = None
        self.rules = []
    
    def _initialize_pipelines(self):
        """Inicializar pipelines com tratamento de erro"""
        try:
            from src.pipelines.data_pipeline import DataPipelineSQL
            from src.pipelines.ml_pipeline import MLPipeline
            from src.pipelines.analysis_pipeline import AnalysisPipeline
            from src.pipelines.performance_pipeline import PerformancePipeline
            
            self.data_pipeline = DataPipelineSQL(force_migration=self.force_migration)
            self.ml_pipeline = MLPipeline()
            self.analysis_pipeline = AnalysisPipeline()
            self.performance_pipeline = PerformancePipeline()
            
            self.logger.info("✅ Pipelines inicializados com sucesso")
            
        except ImportError as e:
            self.logger.error(f"❌ Erro ao importar pipelines: {e}")
            self.logger.error("💡 Verifique se todos os módulos estão implementados")
            raise
        except Exception as e:
            self.logger.error(f"❌ Erro na inicialização dos pipelines: {e}")
            raise
    
    def _check_database_config(self) -> bool:
        """Verificar configuração obrigatória do banco"""
        required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ Variáveis de ambiente ausentes: {missing_vars}")
            else:
                print(f"❌ Variáveis de ambiente ausentes: {missing_vars}")
            
            print("💡 Configure as variáveis de ambiente:")
            print("   export DB_HOST=localhost")
            print("   export DB_NAME=salary_analysis")
            print("   export DB_USER=salary_user")
            print("   export DB_PASSWORD=senha_forte")
            print("🔧 Ou crie arquivo .env com essas variáveis")
            return False
        
        if hasattr(self, 'logger'):
            self.logger.info("✅ Configuração de banco de dados encontrada")
        return True
    
    def run(self):
        """Executar pipeline completo SQL-only com tratamento robusto de erros"""
        self.logger.info("🚀 Sistema de Análise Salarial - VERSÃO SQL EXCLUSIVA")
        self.logger.info("🗄️ Fonte de dados: BANCO DE DADOS MySQL")
        self.logger.info("="*60)
        
        try:
            # 0. Configurações iniciais
            self._setup()
            
            # 1. Pipeline de dados SQL
            self.logger.info("\n📊 PIPELINE DE DADOS SQL")
            self.logger.info("-" * 40)
            self.df = self.data_pipeline.run()
            
            if self.df is None or len(self.df) == 0:
                raise ValueError("❌ Nenhum dado carregado do banco")
            
            self.logger.info(f"✅ Dados carregados: {len(self.df)} registros")
            
            # 2. Pipeline de ML com tratamento de erro
            self.logger.info("\n🤖 PIPELINE DE ML")
            self.logger.info("-" * 40)
            try:
                self.models, self.results = self.ml_pipeline.run(self.df)
                
                if not self.models:
                    self.logger.warning("⚠️ Nenhum modelo foi treinado com sucesso")
                else:
                    self.logger.info(f"✅ {len(self.models)} modelos treinados")
                    
            except Exception as e:
                self.logger.error(f"❌ Erro no pipeline ML: {e}")
                self.logger.warning("⚠️ Continuando sem modelos ML...")
                self.models = {}
                self.results = {}
            
            # 3. Pipeline de performance (apenas se houver modelos)
            if self.models:
                self.logger.info("\n📈 PIPELINE DE PERFORMANCE")
                self.logger.info("-" * 40)
                try:
                    self.performance_pipeline.run(self.models, self.results, self.df)
                except Exception as e:
                    self.logger.error(f"❌ Erro no pipeline de performance: {e}")
            else:
                self.logger.info("\n⚠️ Pulando pipeline de performance (sem modelos)")
            
            # 4. Pipelines de análise
            self.logger.info("\n🎯 PIPELINE DE ANÁLISES")
            self.logger.info("-" * 40)
            
            # Clustering
            try:
                self.best_k = self.analysis_pipeline.run_clustering(self.df)
                if self.best_k:
                    self.logger.info(f"✅ Clustering: {self.best_k} clusters identificados")
            except Exception as e:
                self.logger.error(f"❌ Erro no clustering: {e}")
                self.best_k = None
            
            # Regras de associação
            try:
                self.rules = self.analysis_pipeline.run_association_rules(self.df)
                if self.rules:
                    self.logger.info(f"✅ {len(self.rules)} regras de associação encontradas")
            except Exception as e:
                self.logger.error(f"❌ Erro nas regras de associação: {e}")
                self.rules = []
            
            # Métricas avançadas
            try:
                self.analysis_pipeline.run_advanced_metrics(self.df, self.results)
                self.logger.info("✅ Métricas avançadas calculadas")
            except Exception as e:
                self.logger.error(f"❌ Erro nas métricas avançadas: {e}")
            
            # 5. Criar views SQL de análise
            self.logger.info("\n🗄️ CRIANDO VIEWS DE ANÁLISE")
            self.logger.info("-" * 40)
            try:
                self.data_pipeline.create_analysis_views()
                self.logger.info("✅ Views SQL de análise criadas")
            except Exception as e:
                self.logger.error(f"❌ Erro ao criar views SQL: {e}")
            
            # 6. Relatório final
            self._generate_final_report()
            
            self.logger.info("\n🎉 Pipeline SQL concluído com sucesso!")
            self.logger.info("📊 Para visualizar: streamlit run app.py")
            
        except Exception as e:
            self.logger.error(f"❌ Erro crítico durante execução: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Relatório de emergência
            self._emergency_report()
            raise
    
    def _setup(self):
        """Configurações iniciais SQL-only"""
        try:
            # Configurar estilo de visualização
            setup_matplotlib_style()
            
            # Testar conexão com banco
            self._test_database_connection()
            
            self.logger.info("✅ Configurações SQL iniciais concluídas")
            
        except Exception as e:
            self.logger.error(f"❌ Erro na configuração: {e}")
            raise
    
    def _test_database_connection(self):
        """Testar conexão com banco de dados"""
        try:
            from src.database.connection import DatabaseConnection
            
            with DatabaseConnection() as db:
                result = db.execute_query("SELECT 1 as test")
                if result:
                    self.logger.info("✅ Conexão com banco de dados funcionando")
                else:
                    raise ValueError("Teste de conexão falhou")
                    
        except Exception as e:
            self.logger.error(f"❌ Erro na conexão com banco: {e}")
            self.logger.error("💡 Verifique se MySQL está rodando e credenciais estão corretas")
            raise
    
    def _generate_final_report(self):
        """Gerar relatório final SQL"""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 RELATÓRIO FINAL DO PIPELINE SQL")
        self.logger.info("="*60)
        
        # Estatísticas do banco
        try:
            if hasattr(self.data_pipeline, 'get_statistics_from_sql'):
                stats = self.data_pipeline.get_statistics_from_sql()
                total_records = stats.get('total_records', 0)
                self.logger.info(f"📋 Registros no banco: {total_records:,}")
                
                if 'salary_stats' in stats:
                    self.logger.info("💰 Distribuição salarial:")
                    for salary_stat in stats['salary_stats']:
                        salary_range = salary_stat.get('salary_range', 'N/A')
                        count = salary_stat.get('count', 0)
                        avg_age = salary_stat.get('avg_age', 0)
                        self.logger.info(f"   • {salary_range}: {count:,} pessoas (idade média: {avg_age:.1f})")
        
        except Exception as e:
            self.logger.warning(f"⚠️ Erro nas estatísticas SQL: {e}")
        
        # Dados processados
        if self.df is not None:
            self.logger.info(f"📊 Dataset ML: {len(self.df)} registros processados")
            self.logger.info(f"📋 Colunas: {list(self.df.columns)}")
        
        # Modelos treinados
        if self.results:
            self.logger.info("🤖 Modelos treinados:")
            for name, result in self.results.items():
                accuracy = result.get('accuracy', 0)
                self.logger.info(f"   • {name}: Acurácia = {accuracy:.4f}")
        else:
            self.logger.info("🤖 Nenhum modelo ML foi treinado")
        
        # Clustering
        if self.best_k:
            self.logger.info(f"🎯 Clustering: {self.best_k} clusters identificados")
        else:
            self.logger.info("🎯 Clustering: Não executado")
        
        # Regras de associação
        rules_count = len(self.rules) if self.rules else 0
        self.logger.info(f"📋 Regras de associação: {rules_count} encontradas")
        
        # Arquivos gerados
        self._show_generated_files()
        
        # Instruções finais
        self.logger.info("\n💡 Próximos passos:")
        self.logger.info("   • Dashboard: streamlit run app.py")
        self.logger.info("   • Análises SQL: Use views criadas no banco")
        self.logger.info("   • Modelos salvos em: data/processed/")
        self.logger.info("   • Visualizações em: output/images/")
    
    def _show_generated_files(self):
        """Mostrar arquivos gerados"""
        # Verificar diretório de output
        output_dir = Path("output")
        if output_dir.exists():
            # Imagens
            images_dir = output_dir / "images"
            if images_dir.exists():
                image_files = list(images_dir.glob("*.png"))
                self.logger.info(f"🎨 {len(image_files)} visualizações em output/images/")
            
            # Análises
            analysis_dir = output_dir / "analysis"
            if analysis_dir.exists():
                analysis_files = list(analysis_dir.glob("*"))
                self.logger.info(f"📊 {len(analysis_files)} análises em output/analysis/")
        
        # Modelos salvos
        processed_dir = Path("data/processed")
        if processed_dir.exists():
            model_files = list(processed_dir.glob("*.joblib"))
            self.logger.info(f"🤖 {len(model_files)} modelos salvos em data/processed/")
    
    def _emergency_report(self):
        """Relatório de emergência quando pipeline falha"""
        self.logger.info("\n🚨 RELATÓRIO DE EMERGÊNCIA")
        self.logger.info("="*40)
        
        if self.df is not None:
            self.logger.info(f"📊 Dados carregados: {len(self.df)} registros")
        else:
            self.logger.info("❌ Nenhum dado foi carregado")
        
        self.logger.info(f"🤖 Modelos criados: {len(self.models)}")
        self.logger.info(f"📊 Resultados: {len(self.results)}")
        
        self.logger.info("\n💡 Para diagnóstico:")
        self.logger.info("   • Verificar logs detalhados acima")
        self.logger.info("   • Testar conexão: mysql -u salary_user -p salary_analysis")
        self.logger.info("   • Verificar estrutura: python main.py --setup-db")

def main():
    """Função principal SQL-only"""
    parser = argparse.ArgumentParser(description="Sistema de Análise Salarial SQL-Only")
    parser.add_argument('--migrate', action='store_true', 
                       help='Forçar migração CSV→SQL (se CSV disponível)')
    parser.add_argument('--setup-db', action='store_true',
                       help='Apenas configurar estrutura do banco')
    
    args = parser.parse_args()
    
    # Verificar variáveis de ambiente primeiro
    required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ ERRO: Configuração de banco de dados obrigatória!")
        print(f"❌ Variáveis ausentes: {missing_vars}")
        print("\n💡 Configure as variáveis de ambiente:")
        print("export DB_HOST=localhost")
        print("export DB_NAME=salary_analysis")
        print("export DB_USER=salary_user")
        print("export DB_PASSWORD=senha_forte")
        print("\n🔧 Ou crie arquivo .env com essas variáveis")
        print("\n📋 Exemplo de .env:")
        print("DB_HOST=localhost")
        print("DB_NAME=salary_analysis")
        print("DB_USER=salary_user")
        print("DB_PASSWORD=senha_forte")
        return
    
    # Setup apenas da estrutura do banco
    if args.setup_db:
        try:
            print("🔧 Configurando estrutura do banco de dados...")
            from src.database.migration import DatabaseMigrator
            
            migrator = DatabaseMigrator()
            if migrator.create_database_structure():
                print("✅ Estrutura do banco criada com sucesso!")
                print("💡 Agora execute: python main.py")
            else:
                print("❌ Erro ao criar estrutura do banco")
                
        except ImportError as e:
            print(f"❌ Erro ao importar DatabaseMigrator: {e}")
            print("💡 Verifique se o módulo src.database.migration existe")
        except Exception as e:
            print(f"❌ Erro: {e}")
        return
    
    # Executar pipeline principal
    try:
        print("🚀 Iniciando Sistema de Análise Salarial...")
        pipeline = MasterPipelineSQL(force_migration=args.migrate)
        pipeline.run()
        
        print("\n🎉 Pipeline executado com sucesso!")
        print("📊 Para visualizar resultados: streamlit run app.py")
        
    except ValueError as e:
        print(f"\n❌ ERRO DE CONFIGURAÇÃO: {e}")
        print("\n💡 Soluções:")
        print("  1. Verificar variáveis de ambiente")
        print("  2. Criar estrutura: python main.py --setup-db")
        print("  3. Testar conexão: mysql -u salary_user -p salary_analysis")
        
    except ImportError as e:
        print(f"\n❌ ERRO DE IMPORTAÇÃO: {e}")
        print("\n💡 Soluções:")
        print("  1. Verificar se todos os módulos existem em src/")
        print("  2. Verificar dependências: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {e}")
        print("\n💡 Soluções possíveis:")
        print("  1. Verificar conexão: mysql -u salary_user -p salary_analysis")
        print("  2. Recriar estrutura: python main.py --setup-db")
        print("  3. Migrar dados: python main.py --migrate")
        print("  4. Verificar logs detalhados nos arquivos de log")

if __name__ == "__main__":
    main()