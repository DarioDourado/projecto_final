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
        'scikit-learn', 'mlxtend'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Pacotes faltantes: {', '.join(missing_packages)}")
        print("Execute: pip install -r requirements.txt")
        return False
    
    print("✅ Todas as dependências estão instaladas!")
    return True


def create_directories():
    """Criar diretórios necessários"""
    print("📁 Criando diretórios...")
    
    directories = [
        "config",
        "translate", 
        "data/raw",
        "data/processed",
        "output/images",
        "output/analysis",
        "output/reports",
        "output/logs",
        "src/utils",
        "src/auth",
        "src/components",
        "src/pages"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")


def run_dashboard():
    """Executar o dashboard"""
    print("🚀 Iniciando Dashboard Multilingual...")
    
    # Verificar se o arquivo principal existe
    main_file = "app_multilingual.py"
    
    if not Path(main_file).exists():
        print(f"❌ Arquivo {main_file} não encontrado!")
        return False
    
    # Executar Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", main_file,
            "--server.port", "8501",
            "--server.address", "localhost"
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
    print("📊 Versão 4.0 - Enhanced & Multilingual")
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