"""Diagnóstico rápido do sistema"""

import sys
import os
from pathlib import Path

def diagnose():
    """Diagnosticar sistema atual"""
    # Voltar para diretório raiz se executado da pasta setup_scripts
    if Path.cwd().name == "setup_scripts":
        os.chdir("..")
    
    print("🔍 DIAGNÓSTICO DO SISTEMA")
    print("=" * 30)
    
    # 1. Python
    print(f"🐍 Python: {sys.version}")
    
    # 2. Dependências críticas
    deps = [
        'mysql.connector',
        'pandas', 
        'numpy',
        'sklearn',
        'streamlit'
    ]
    
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep}")
    
    # 3. Arquivos importantes
    files = ['.env', 'main.py', 'app.py']
    for file in files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
    
    # 4. Estrutura
    dirs = ['src/database', 'src/pipelines', 'data', 'output']
    for directory in dirs:
        if Path(directory).exists():
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/")
    
    # 5. MySQL
    try:
        import mysql.connector
        print("✅ MySQL driver disponível")
        
        # Testar conexão básica
        try:
            conn = mysql.connector.connect(
                host='localhost',
                user='root',
                password='',
                connection_timeout=3
            )
            print("✅ MySQL acessível")
            conn.close()
        except:
            print("⚠️  MySQL não acessível")
    except ImportError:
        print("❌ MySQL driver ausente")

if __name__ == "__main__":
    diagnose()