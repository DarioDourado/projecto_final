"""
🗄️ Setup de Banco de Dados
Configuração e inicialização do MySQL
"""

import logging
import os
from pathlib import Path
from typing import Optional

def setup_database() -> bool:
    """
    Configurar banco de dados MySQL
    
    Returns:
        bool: True se configuração foi bem-sucedida
    """
    logger = logging.getLogger(__name__)
    
    print("🗄️ CONFIGURAÇÃO DO BANCO DE DADOS")
    print("=" * 50)
    
    try:
        # 1. Verificar configurações
        if not _check_database_config():
            print("❌ Configurações de banco não encontradas")
            _show_config_instructions()
            return False
        
        # 2. Testar conexão
        print("📋 Testando conexão com banco...")
        if not _test_connection():
            print("❌ Não foi possível conectar ao banco")
            return False
        
        print("✅ Conexão com banco OK")
        
        # 3. Criar estrutura
        print("🔧 Criando estrutura do banco...")
        if not _create_database_structure():
            print("❌ Erro ao criar estrutura")
            return False
        
        print("✅ Estrutura do banco criada")
        
        # 4. Executar migração (se necessário)
        print("🔄 Verificando necessidade de migração...")
        if _needs_migration():
            print("📊 Executando migração de dados...")
            if not _run_migration():
                print("⚠️ Problemas na migração, mas banco está configurado")
                return True
            print("✅ Migração concluída")
        else:
            print("ℹ️ Migração não necessária")
        
        print("\n🎉 BANCO DE DADOS CONFIGURADO COM SUCESSO!")
        return True
        
    except Exception as e:
        logger.error(f"Erro na configuração do banco: {e}")
        print(f"❌ Erro na configuração: {e}")
        return False

def _check_database_config() -> bool:
    """Verificar se as configurações de banco estão presentes"""
    required_vars = ['DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME']
    
    # Verificar variáveis de ambiente
    env_vars = {var: os.getenv(var) for var in required_vars}
    
    # Verificar arquivo .env
    env_file = Path('.env')
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                env_content = f.read()
                for var in required_vars:
                    if f"{var}=" in env_content and not env_vars[var]:
                        # Extrair valor do .env
                        for line in env_content.split('\n'):
                            if line.startswith(f"{var}="):
                                env_vars[var] = line.split('=', 1)[1].strip()
        except Exception:
            pass
    
    # Verificar se todas as variáveis estão definidas
    missing_vars = [var for var, value in env_vars.items() if not value]
    
    if missing_vars:
        print(f"⚠️ Variáveis de ambiente em falta: {missing_vars}")
        return False
    
    print("✅ Configurações de banco encontradas")
    return True

def _show_config_instructions():
    """Mostrar instruções de configuração"""
    print("\n💡 INSTRUÇÕES DE CONFIGURAÇÃO:")
    print("1. Crie um arquivo .env na raiz do projeto com:")
    print("""
DB_HOST=localhost
DB_USER=salary_user
DB_PASSWORD=sua_senha_aqui
DB_NAME=salary_analysis
""")
    print("2. Ou defina as variáveis de ambiente:")
    print("   export DB_HOST=localhost")
    print("   export DB_USER=salary_user")
    print("   export DB_PASSWORD=sua_senha")
    print("   export DB_NAME=salary_analysis")
    print("\n3. Certifique-se de que o MySQL está rodando")

def _test_connection() -> bool:
    """Testar conexão com o banco"""
    try:
        from src.database.connection import create_connection
        
        conn = create_connection()
        if conn:
            conn.close()
            return True
        return False
        
    except ImportError:
        print("⚠️ Módulo de conexão não encontrado")
        return False
    except Exception as e:
        print(f"❌ Erro na conexão: {e}")
        return False

def _create_database_structure() -> bool:
    """Criar estrutura do banco de dados"""
    try:
        from src.database.migration import create_tables
        return create_tables()
        
    except ImportError:
        print("⚠️ Módulo de migração não encontrado")
        return False
    except Exception as e:
        print(f"❌ Erro ao criar estrutura: {e}")
        return False

def _needs_migration() -> bool:
    """Verificar se precisa de migração de dados"""
    try:
        from src.database.models import SalaryAnalysisSQL
        
        sql_model = SalaryAnalysisSQL()
        df = sql_model.get_dataset_for_ml()
        
        # Se não há dados ou muito poucos, precisa migrar
        return df is None or len(df) < 100
        
    except Exception:
        return True  # Em caso de erro, assumir que precisa migrar

def _run_migration() -> bool:
    """Executar migração de dados CSV → SQL"""
    try:
        from src.database.migration import migrate_csv_to_sql
        return migrate_csv_to_sql()
        
    except ImportError:
        print("⚠️ Sistema de migração não disponível")
        return False
    except Exception as e:
        print(f"❌ Erro na migração: {e}")
        return False

if __name__ == "__main__":
    success = setup_database()
    if success:
        print("\n✅ Setup concluído com sucesso!")
    else:
        print("\n❌ Setup falhou. Sistema funcionará em modo CSV.")