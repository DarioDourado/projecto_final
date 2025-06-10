#!/usr/bin/env python3
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
    
    print("\n🎯 Sistema testado!")

if __name__ == "__main__":
    test_system()
