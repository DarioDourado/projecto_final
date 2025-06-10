#!/usr/bin/env python3
"""
🚀 Script de Inicialização do Dashboard Multilingual
Executa verificações e inicia o sistema
"""

import sys
import subprocess
import os
from pathlib import Path


def check_requirements():
    """Verificar se todas as dependências estão instaladas"""
    print("🔍 Verificando dependências...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'mysql-connector-python', 'scikit-learn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Dependências em falta: {missing}")
        print("💡 Execute: pip install -r requirements.txt")
        return False
    
    print("✅ Todas as dependências estão instaladas!")
    return True


def create_directories():
    """Criar diretórios necessários"""
    print("📁 Criando diretórios...")
    
    directories = [
        "config", "logs", "translate", "data/raw", "data/processed",
        "output/images", "output/analysis", "models", "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")


def run_dashboard():
    """Executar o dashboard"""
    print("🚀 Iniciando Dashboard Multilingual...")
    
    # ✅ ATUALIZADO: Usar app.py como arquivo principal
    main_file = "app.py"
    
    if not Path(main_file).exists():
        print(f"❌ Arquivo {main_file} não encontrado!")
        return False
    
    # Executar Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", main_file,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard encerrado pelo usuário")
    except Exception as e:
        print(f"❌ Erro ao executar dashboard: {e}")
        return False
    
    return True


def main():
    """Função principal"""
    print("=" * 60)
    print("🌍 DASHBOARD MULTILINGUAL - ANÁLISE SALARIAL")
    print("📊 Versão 6.1 - Sistema Modular Consolidado")
    print("=" * 60)
    
    # Verificações
    if not check_requirements():
        sys.exit(1)
    
    create_directories()
    
    print("\n🎯 Tudo pronto! Iniciando dashboard...")
    print("📱 Acesse: http://localhost:8501")
    print("🔄 Para parar: Ctrl+C")
    print("-" * 60)
    
    # Executar dashboard
    run_dashboard()


if __name__ == "__main__":
    main()