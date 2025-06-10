"""Setup completo para sistema SQL-only - Versão Robusta"""

import os
import sys
from pathlib import Path

# Adicionar src ao path primeiro
sys.path.append(str(Path(__file__).parent / "src"))

def check_and_install_dependencies():
    """Verificar e orientar sobre instalação de dependências"""
    print("🔍 Verificando dependências...")
    
    missing_deps = []
    
    # Verificar dependências essenciais
    dependencies = [
        ('mysql.connector', 'mysql-connector-python'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('pathlib', 'pathlib2'),
        ('dotenv', 'python-dotenv')
    ]
    
    for import_name, package_name in dependencies:
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name}")
            missing_deps.append(package_name)
    
    if missing_deps:
        print(f"\n⚠️  Dependências ausentes: {missing_deps}")
        print("\n💡 Para instalar:")
        print("   pip install mysql-connector-python python-dotenv pandas numpy")
        print("   # OU")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ Todas as dependências estão instaladas")
    return True

def load_environment():
    """Carregar variáveis de ambiente do .env se disponível"""
    env_path = Path(".env")
    
    if env_path.exists():
        try:
            # Tentar usar python-dotenv se disponível
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ Variáveis carregadas do arquivo .env")
            return True
        except ImportError:
            print("⚠️  python-dotenv não disponível, lendo .env manualmente")
            
            # Ler .env manualmente se python-dotenv não estiver disponível
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                print("✅ Variáveis carregadas manualmente do .env")
                return True
            except Exception as e:
                print(f"⚠️  Erro ao ler .env: {e}")
                return False
    else:
        print("ℹ️  Arquivo .env não encontrado, usando variáveis de ambiente do sistema")
        return True

def setup_database():
    """Setup completo do banco de dados"""
    
    print("🗄️ Setup Sistema SQL-Only - Versão Robusta")
    print("=" * 50)
    
    # 1. Verificar dependências
    if not check_and_install_dependencies():
        return False
    
    # 2. Carregar ambiente
    load_environment()
    
    # 3. Verificar configuração
    required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
    
    print("\n1️⃣ Verificando configuração...")
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mascarar senha para segurança
            display_value = "*" * len(value) if var == 'DB_PASSWORD' else value
            print(f"   ✅ {var}: {display_value}")
        else:
            print(f"   ❌ {var}: não configurado")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n❌ Variáveis ausentes: {missing_vars}")
        print("💡 Soluções:")
        print("   1. python setup_sql_only.py --create-env  (criar arquivo .env)")
        print("   2. export DB_HOST=localhost  (configurar manualmente)")
        print("   3. Editar arquivo .env existente")
        return False
    
    # 4. Testar conexão
    print("\n2️⃣ Testando conexão com MySQL...")
    try:
        import mysql.connector
        from mysql.connector import Error
        
        config = {
            'host': os.getenv('DB_HOST'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'charset': 'utf8mb4'
        }
        
        conn = mysql.connector.connect(**config)
        if conn.is_connected():
            db_info = conn.get_server_info()
            print(f"   ✅ Conexão estabelecida (MySQL {db_info})")
            conn.close()
        else:
            print("   ❌ Falha na conexão")
            return False
            
    except Error as e:
        print(f"   ❌ Erro de conexão MySQL: {e}")
        print("💡 Verifique:")
        print("   • MySQL está rodando?")
        print("   • Credenciais estão corretas?")
        print("   • Database existe?")
        return False
    except ImportError as e:
        print(f"   ❌ Dependência ausente: {e}")
        print("💡 Instale: pip install mysql-connector-python")
        return False
    
    # 5. Criar estrutura do banco
    print("\n3️⃣ Criando estrutura do banco...")
    try:
        from src.database.migration import DatabaseMigrator
        
        migrator = DatabaseMigrator()
        if migrator.create_database_structure():
            print("   ✅ Estrutura do banco criada com sucesso")
        else:
            print("   ❌ Erro ao criar estrutura do banco")
            return False
    except ImportError as e:
        print(f"   ❌ Módulo não encontrado: {e}")
        print("💡 Verifique se a estrutura src/database/ existe")
        return False
    except Exception as e:
        print(f"   ❌ Erro inesperado: {e}")
        return False
    
    # 6. Verificar dados existentes
    print("\n4️⃣ Verificando dados no banco...")
    try:
        from src.database.models import SalaryAnalysisSQL
        
        sql_model = SalaryAnalysisSQL()
        stats = sql_model.get_advanced_statistics()
        total = stats.get('total_records', 0)
        
        if total > 0:
            print(f"   ✅ {total:,} registros encontrados no banco")
            
            # Mostrar distribuição por salário se disponível
            if 'salary_stats' in stats:
                print("   📊 Distribuição salarial:")
                for salary_stat in stats['salary_stats']:
                    range_name = salary_stat['salary_range']
                    count = salary_stat['count']
                    print(f"      • {range_name}: {count:,} registros")
        else:
            print("   ⚠️  Banco está vazio")
            print("   💡 Execute migração: python main.py --migrate")
            
            # Verificar se CSV existe para migração
            csv_paths = [
                Path("data/raw/4-Carateristicas_salario.csv"),
                Path("bkp/4-Carateristicas_salario.csv"),
                Path("4-Carateristicas_salario.csv")
            ]
            
            csv_found = None
            for path in csv_paths:
                if path.exists():
                    csv_found = path
                    break
            
            if csv_found:
                print(f"   📄 CSV encontrado: {csv_found}")
                print("   💡 Para migrar automaticamente: python main.py --migrate")
            else:
                print("   ⚠️  Nenhum CSV encontrado para migração")
                print("   💡 Coloque o arquivo CSV em uma das localizações:")
                for path in csv_paths:
                    print(f"      • {path}")
            
    except ImportError as e:
        print(f"   ❌ Módulo não encontrado: {e}")
        print("💡 Verifique se src/database/models.py existe")
    except Exception as e:
        print(f"   ⚠️  Não foi possível verificar dados: {e}")
    
    # 7. Verificar estrutura de arquivos
    print("\n5️⃣ Verificando estrutura do projeto...")
    required_dirs = [
        "src/database",
        "src/pipelines", 
        "src/utils",
        "data",
        "output"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path}")
            print(f"      💡 Crie: mkdir -p {dir_path}")
    
    # 8. Relatório final
    print("\n" + "=" * 50)
    print("✅ Setup SQL-Only concluído!")
    print("\n🚀 Próximos passos:")
    print("   1. python main.py                 # Executar pipeline completo")
    print("   2. python main.py --migrate       # Migrar dados CSV (se necessário)")
    print("   3. streamlit run app.py           # Abrir dashboard")
    print("   4. python main.py --setup-db      # Apenas recriar estrutura")
    
    print("\n📚 Comandos úteis:")
    print("   • Testar conexão: python -c \"from src.database.connection import DatabaseConnection; print('OK' if DatabaseConnection().connect() else 'ERRO')\"")
    print("   • Ver logs: tail -f logs/app.log")
    print("   • Backup: mysqldump -u $(echo $DB_USER) -p$(echo $DB_PASSWORD) $(echo $DB_NAME) > backup.sql")
    
    return True

def create_env_file():
    """Criar arquivo .env de exemplo com configurações detalhadas"""
    env_content = """# =================================================
# Configuração do Sistema SQL-Only
# =================================================

# Banco de Dados MySQL (OBRIGATÓRIO)
DB_HOST=localhost
DB_NAME=salary_analysis
DB_USER=salary_user
DB_PASSWORD=senha_forte

# Configurações do Sistema
USE_DATABASE=true
AUTO_MIGRATE=true

# Configurações de Log
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Configurações de ML
RANDOM_STATE=42
TEST_SIZE=0.2

# Configurações de Performance
CACHE_TTL=300
MAX_WORKERS=4

# Configurações de Segurança (PRODUÇÃO)
# DB_SSL_MODE=REQUIRED
# DB_SSL_CERT=path/to/cert.pem
# DB_SSL_KEY=path/to/key.pem
# DB_SSL_CA=path/to/ca.pem

# =================================================
# Para usar em produção, configure as variáveis
# de ambiente ao invés de usar este arquivo
# =================================================
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text(env_content)
        print(f"✅ Arquivo .env criado: {env_path}")
        print("🔧 Edite as configurações do banco de dados conforme necessário")
        print("\n📝 Configurações obrigatórias a editar:")
        print("   • DB_HOST (seu servidor MySQL)")
        print("   • DB_NAME (nome do banco)")
        print("   • DB_USER (usuário MySQL)")
        print("   • DB_PASSWORD (senha do usuário)")
    else:
        print(f"⚠️  Arquivo .env já existe: {env_path}")
        print("💡 Para recriar: rm .env && python setup_sql_only.py --create-env")

def check_mysql_service():
    """Verificar se MySQL está rodando"""
    print("\n🔍 Verificando serviço MySQL...")
    
    try:
        import mysql.connector
        from mysql.connector import Error
        
        # Tentar conectar sem especificar database
        basic_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        
        conn = mysql.connector.connect(**basic_config)
        if conn.is_connected():
            print("   ✅ MySQL está rodando")
            
            # Verificar se database existe
            cursor = conn.cursor()
            db_name = os.getenv('DB_NAME', 'salary_analysis')
            cursor.execute("SHOW DATABASES")
            databases = [db[0] for db in cursor.fetchall()]
            
            if db_name in databases:
                print(f"   ✅ Database '{db_name}' existe")
            else:
                print(f"   ⚠️  Database '{db_name}' não existe")
                print(f"   💡 Crie com: CREATE DATABASE {db_name};")
            
            conn.close()
            return True
    except ImportError:
        print("   ❌ mysql-connector-python não instalado")
        print("   💡 Instale: pip install mysql-connector-python")
        return False
    except Error as e:
        print(f"   ❌ MySQL não está acessível: {e}")
        print("   💡 Comandos para iniciar MySQL:")
        print("      • Linux: sudo systemctl start mysql")
        print("      • macOS: brew services start mysql")
        print("      • Windows: net start mysql")
        return False

def create_project_structure():
    """Criar estrutura básica do projeto se não existir"""
    print("\n📁 Criando estrutura do projeto...")
    
    directories = [
        "src/database",
        "src/pipelines",
        "src/utils",
        "src/config",
        "data/raw",
        "data/processed",
        "output/images",
        "output/analysis",
        "logs",
        "models/trained"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Criado: {directory}")
        else:
            print(f"   ✅ Existe: {directory}")

def main():
    """Função principal com opções"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--create-env":
            create_env_file()
        elif sys.argv[1] == "--check-mysql":
            load_environment()
            check_mysql_service()
        elif sys.argv[1] == "--create-structure":
            create_project_structure()
        elif sys.argv[1] == "--help":
            print("🗄️ Setup SQL-Only - Opções:")
            print("  python setup_sql_only.py                   # Setup completo")
            print("  python setup_sql_only.py --create-env      # Criar arquivo .env")
            print("  python setup_sql_only.py --check-mysql     # Verificar MySQL")
            print("  python setup_sql_only.py --create-structure # Criar diretórios")
            print("  python setup_sql_only.py --help            # Esta ajuda")
        else:
            print(f"❌ Opção desconhecida: {sys.argv[1]}")
            print("💡 Use --help para ver opções disponíveis")
    else:
        setup_database()

if __name__ == "__main__":
    main()