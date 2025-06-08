"""Script para executar apenas a análise exploratória"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main import run_eda

if __name__ == "__main__":
    run_eda()