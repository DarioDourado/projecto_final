"""Configuração Completa Automática do Sistema - Versão Final"""

import subprocess
import sys
import os
from pathlib import Path
import time

def check_python_version():
    """Verificar versão do Python"""
    print("🐍 Verificando versão do Python...")
    
    if sys.version_info < (3, 8):
        print(f"❌ Python {sys.version} é muito antigo")
        print("💡 Requerido: Python 3.8+")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def install_dependencies():
    """Instalar todas as dependências necessárias"""
    print("\n📦 Instalando dependências...")
    
    dependencies = [
        'mysql-connector-python',
        'python-dotenv',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'streamlit',
        'plotly',
        'joblib',
        'mlxtend'
    ]
    
    for dep in dependencies:
        try:
            print(f"   📦 Instalando {dep}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', dep
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {dep} instalado")
        except subprocess.CalledProcessError:
            print(f"   ❌ Erro ao instalar {dep}")
            return False
    
    print("✅ Todas as dependências instaladas!")
    return True

def test_imports():
    """Testar imports críticos"""
    print("\n🔍 Testando imports...")
    
    imports_test = [
        ('mysql.connector', 'mysql-connector-python'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('streamlit', 'streamlit'),
        ('dotenv', 'python-dotenv')
    ]
    
    for module, package in imports_test:
        try:
            __import__(module)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            return False
    
    print("✅ Todos os imports funcionando!")
    return True

def create_project_structure():
    """Criar estrutura completa do projeto"""
    print("\n📁 Criando estrutura do projeto...")
    
    directories = [
        "src/database",
        "src/pipelines", 
        "src/utils",
        "src/visualization",
        "src/config",
        "data/raw",
        "data/processed",
        "output/images",
        "output/analysis",
        "logs",
        "models/trained",
        "models/backup",
        "bkp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    # Criar __init__.py necessários
    init_files = [
        "src/__init__.py",
        "src/database/__init__.py",
        "src/pipelines/__init__.py",
        "src/utils/__init__.py",
        "src/visualization/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
    
    print("✅ Estrutura do projeto criada!")
    return True

def check_mysql_installation():
    """Verificar se MySQL está instalado"""
    print("\n🗄️ Verificando MySQL...")
    
    # Verificar se mysql command existe
    try:
        result = subprocess.run(['mysql', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ MySQL instalado")
            
            # Verificar se está rodando
            try:
                subprocess.run(['pgrep', 'mysql'], 
                              capture_output=True, check=True)
                print("   ✅ MySQL está rodando")
                return True
            except subprocess.CalledProcessError:
                print("   ⚠️  MySQL não está rodando")
                print("   💡 Inicie: brew services start mysql (macOS)")
                print("   💡 Inicie: sudo systemctl start mysql (Linux)")
                return False
        else:
            print("   ❌ MySQL não encontrado")
            return False
    except FileNotFoundError:
        print("   ❌ MySQL não instalado")
        print("   💡 Instale: brew install mysql (macOS)")
        print("   💡 Instale: sudo apt install mysql-server (Linux)")
        return False

def create_env_file():
    """Criar arquivo .env completo"""
    print("\n📄 Criando arquivo .env...")
    
    env_content = """# =================================================
# Configuração do Sistema de Análise Salarial
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

# Configurações de Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true

# =================================================
# CONFIGURE AS CREDENCIAIS DO MYSQL ACIMA
# =================================================
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text(env_content)
        print("   ✅ Arquivo .env criado")
    else:
        print("   ✅ Arquivo .env já existe")
    
    return True

def create_minimal_modules():
    """Criar módulos mínimos necessários"""
    print("\n🔧 Criando módulos básicos...")
    
    # src/utils/logger.py
    logger_content = '''"""Sistema de logging"""
import logging
from pathlib import Path

def setup_logging():
    """Configurar logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
'''
    
    # src/visualization/styles.py
    styles_content = '''"""Estilos de visualização"""
import matplotlib.pyplot as plt

def setup_matplotlib_style():
    """Configurar estilo matplotlib"""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
'''
    
    # Criar arquivos
    modules = {
        "src/utils/logger.py": logger_content,
        "src/visualization/styles.py": styles_content
    }
    
    for file_path, content in modules.items():
        path = Path(file_path)
        if not path.exists():
            path.write_text(content)
            print(f"   ✅ {file_path}")
    
    return True

def test_mysql_connection():
    """Testar conexão MySQL com credenciais padrão"""
    print("\n🔍 Testando conexão MySQL...")
    
    try:
        import mysql.connector
        
        # Tentar várias combinações
        test_configs = [
            {'user': 'root', 'password': ''},
            {'user': 'root', 'password': 'root'},
            {'user': 'root', 'password': 'password'},
            {'user': 'salary_user', 'password': 'senha_forte', 'database': 'salary_analysis'}
        ]
        
        for config in test_configs:
            try:
                config['host'] = 'localhost'
                conn = mysql.connector.connect(**config)
                if conn.is_connected():
                    print(f"   ✅ Conexão funciona: {config['user']}")
                    conn.close()
                    return config
            except mysql.connector.Error:
                continue
        
        print("   ❌ Nenhuma configuração funcionou")
        return None
        
    except ImportError:
        print("   ❌ mysql-connector-python não disponível")
        return None

def setup_mysql_database(working_config):
    """Configurar database MySQL se possível"""
    if not working_config:
        print("\n⚠️  Pulando configuração MySQL (sem acesso)")
        return False
    
    print("\n🗄️ Configurando database MySQL...")
    
    try:
        import mysql.connector
        
        conn = mysql.connector.connect(**working_config)
        cursor = conn.cursor()
        
        # Criar database
        cursor.execute("CREATE DATABASE IF NOT EXISTS salary_analysis CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        
        # Criar usuário se for root
        if working_config['user'] == 'root':
            try:
                cursor.execute("CREATE USER IF NOT EXISTS 'salary_user'@'localhost' IDENTIFIED BY 'senha_forte'")
                cursor.execute("GRANT ALL PRIVILEGES ON salary_analysis.* TO 'salary_user'@'localhost'")
                cursor.execute("FLUSH PRIVILEGES")
                print("   ✅ Usuário salary_user criado")
            except mysql.connector.Error as e:
                print(f"   ⚠️  Usuário já existe ou sem permissão: {e}")
        
        # Verificar database
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall()]
        
        if 'salary_analysis' in databases:
            print("   ✅ Database 'salary_analysis' pronta")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Erro na configuração: {e}")
        return False

def create_test_script():
    """Criar script de teste do sistema"""
    print("\n🧪 Criando script de teste...")
    
    test_content = '''#!/usr/bin/env python3
"""Script de teste do sistema"""

def test_system():
    """Testar todos os componentes"""
    print("🧪 TESTE DO SISTEMA")
    print("=" * 30)
    
    # Teste 1: Imports
    try:
        import mysql.connector
        import pandas as pd
        import numpy as np
        import sklearn
        print("✅ Imports básicos")
    except ImportError as e:
        print(f"❌ Erro de import: {e}")
        return False
    
    # Teste 2: Conexão MySQL
    try:
        from pathlib import Path
        import os
        
        # Carregar .env
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
        
        config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'salary_analysis'),
            'user': os.getenv('DB_USER', 'salary_user'),
            'password': os.getenv('DB_PASSWORD', 'senha_forte')
        }
        
        conn = mysql.connector.connect(**config)
        if conn.is_connected():
            print("✅ Conexão MySQL")
            conn.close()
        
    except Exception as e:
        print(f"❌ Conexão MySQL: {e}")
    
    # Teste 3: Estrutura
    required_dirs = ["src/database", "src/pipelines", "data", "output"]
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory}")
    
    print("\\n🎯 Sistema testado!")

if __name__ == "__main__":
    test_system()
'''
    
    test_file = Path("test_system.py")
    test_file.write_text(test_content)
    test_file.chmod(0o755)
    print("   ✅ test_system.py criado")
    
    return True

def show_next_steps():
    """Mostrar próximos passos"""
    print("\n" + "="*60)
    print("🎉 CONFIGURAÇÃO AUTOMÁTICA CONCLUÍDA!")
    print("="*60)
    
    print("\n🚀 PRÓXIMOS PASSOS:")
    print("1. 🔧 Configure MySQL:")
    print("   • Edite .env com suas credenciais MySQL")
    print("   • Ou use as padrão se funcionarem")
    
    print("\n2. 🧪 Teste o sistema:")
    print("   python test_system.py")
    
    print("\n3. 📊 Configure a estrutura do banco:")
    print("   python main.py --setup-db")
    
    print("\n4. 🗄️ Execute migração (se tiver CSV):")
    print("   python main.py --migrate")
    
    print("\n5. 🚀 Execute o pipeline completo:")
    print("   python main.py")
    
    print("\n6. 🌐 Abra o dashboard:")
    print("   streamlit run app.py")
    
    print("\n📋 ARQUIVOS IMPORTANTES:")
    print("• .env           → Configurações")
    print("• test_system.py → Teste do sistema") 
    print("• main.py        → Pipeline principal")
    print("• app.py         → Dashboard")
    
    print("\n💡 RESOLUÇÃO DE PROBLEMAS:")
    print("• Logs em: logs/app.log")
    print("• Teste: python test_system.py")
    print("• MySQL: mysql -u salary_user -p salary_analysis")

def main():
    """Configuração automática completa"""
    print("🎯 CONFIGURAÇÃO AUTOMÁTICA DO SISTEMA")
    print("🎯 Sistema de Análise Salarial - Setup Completo")
    print("="*60)
    
    start_time = time.time()
    
    # Etapa 1: Python
    if not check_python_version():
        return False
    
    # Etapa 2: Dependências
    if not install_dependencies():
        print("❌ Falha na instalação de dependências")
        return False
    
    # Etapa 3: Testar imports
    if not test_imports():
        print("❌ Falha nos imports")
        return False
    
    # Etapa 4: Estrutura
    if not create_project_structure():
        return False
    
    # Etapa 5: Módulos básicos
    if not create_minimal_modules():
        return False
    
    # Etapa 6: MySQL
    mysql_ok = check_mysql_installation()
    
    # Etapa 7: Arquivo .env
    create_env_file()
    
    # Etapa 8: Testar MySQL
    working_config = test_mysql_connection()
    
    # Etapa 9: Configurar database
    if working_config:
        setup_mysql_database(working_config)
    
    # Etapa 10: Script de teste
    create_test_script()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Configuração concluída em {elapsed:.1f} segundos")
    
    # Próximos passos
    show_next_steps()
    
    return True

if __name__ == "__main__":
    # Ajustar paths para execução da pasta setup_scripts
    os.chdir("..")  # Voltar para diretório raiz do projeto
    
    try:
        success = main()
        if success:
            print("\n✅ Sistema configurado com sucesso!")
        else:
            print("\n❌ Configuração falhou")
    except KeyboardInterrupt:
        print("\n⚠️  Configuração interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro na configuração: {e}")