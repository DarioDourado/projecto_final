"""Setup completo para sistema SQL-only"""

import os
import sys
from pathlib import Path
import mysql.connector
from mysql.connector import Error

def setup_database():
    """Setup completo do banco de dados"""
    
    print("🗄️ Setup Sistema SQL-Only")
    print("=" * 40)
    
    # Verificar configuração
    required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
    
    print("1. Verificando configuração...")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: {value}")
        else:
            print(f"   ❌ {var}: não configurado")
            return False
    
    # Testar conexão
    print("\n2. Testando conexão...")
    try:
        config = {
            'host': os.getenv('DB_HOST'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'charset': 'utf8mb4'
        }
        
        conn = mysql.connector.connect(**config)
        if conn.is_connected():
            print("   ✅ Conexão estabelecida")
            conn.close()
        else:
            print("   ❌ Falha na conexão")
            return False
            
    except Error as e:
        print(f"   ❌ Erro de conexão: {e}")
        return False
    
    # Criar estrutura
    print("\n3. Criando estrutura do banco...")
    try:
        sys.path.append(str(Path(__file__).parent / "src"))
        from src.database.migration import DatabaseMigrator
        
        migrator = DatabaseMigrator()
        if migrator.create_database_structure():
            print("   ✅ Estrutura criada")
        else:
            print("   ❌ Erro na estrutura")
            return False
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return False
    
    # Verificar dados
    print("\n4. Verificando dados...")
    try:
        from src.database.models import SalaryAnalysisSQL
        
        sql_model = SalaryAnalysisSQL()
        stats = sql_model.get_advanced_statistics()
        total = stats.get('total_records', 0)
        
        if total > 0:
            print(f"   ✅ {total:,} registros encontrados")
        else:
            print("   ⚠️ Banco vazio - execute migração se necessário")
            print("   💡 python main.py --migrate")
    except Exception as e:
        print(f"   ⚠️ Não foi possível verificar dados: {e}")
    
    print("\n✅ Setup concluído!")
    print("\n🚀 Próximos passos:")
    print("   • python main.py  (executar pipeline)")
    print("   • streamlit run app.py  (dashboard)")
    
    return True

def create_env_file():
    """Criar arquivo .env de exemplo"""
    env_content = """# Configuração do Banco de Dados
DB_HOST=localhost
DB_NAME=salary_analysis
DB_USER=salary_user
DB_PASSWORD=senha_forte

# Configurações do Sistema
USE_DATABASE=true
AUTO_MIGRATE=true
LOG_LEVEL=INFO
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text(env_content)
        print(f"✅ Arquivo .env criado: {env_path}")
    else:
        print(f"⚠️ Arquivo .env já existe: {env_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--create-env":
        create_env_file()
    else:
        setup_database()