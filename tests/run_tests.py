#!/usr/bin/env python3
"""
🧪 Script de Execução de Testes Completos
Executa todos os testes de Data Science com relatórios detalhados
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_comprehensive_tests():
    """Executar suite completa de testes"""
    print("🧪 EXECUTANDO TESTES COMPLETOS DE DATA SCIENCE")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Comandos de teste
    test_commands = [
        # Testes básicos
        ["pytest", "tests/test_data_pipeline.py", "-v"],
        
        # Testes de ML
        ["pytest", "tests/test_ml_pipeline.py", "-v"],
        
        # Testes de performance
        ["pytest", "tests/test_performance_metrics.py", "-v"],
        
        # Todos os testes com coverage
        ["pytest", "--cov=src", "--cov-report=html", "--cov-report=term-missing"]
    ]
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n🔄 Executando comando {i}/{len(test_commands)}: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Comando {i} executado com sucesso")
            else:
                print(f"⚠️ Comando {i} com warnings:")
                print(result.stdout)
                if result.stderr:
                    print("Erros:", result.stderr)
        
        except Exception as e:
            print(f"❌ Erro no comando {i}: {e}")
    
    print(f"\n✅ TESTES CONCLUÍDOS - {timestamp}")

if __name__ == "__main__":
    run_comprehensive_tests()