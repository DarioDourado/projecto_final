"""Script para executar o dashboard"""

import streamlit as st
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    """Executar dashboard"""
    import subprocess
    
    dashboard_path = Path(__file__).parent.parent / "dashboard_app.py"
    
    if dashboard_path.exists():
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path)
        ])
    else:
        print("❌ Arquivo dashboard_app.py não encontrado")

if __name__ == "__main__":
    main()