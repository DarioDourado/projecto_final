"""Script para configurar estrutura de dados correta"""

import os
import shutil
from pathlib import Path

def setup_data_structure():
    """Configurar estrutura de pastas e mover arquivos se necessário"""
    project_root = Path(__file__).parent
    
    # Criar estrutura de pastas
    folders_to_create = [
        "data/raw",
        "data/processed", 
        "output/images",
        "output/models",
        "output/logs",
        "models/trained"
    ]
    
    for folder in folders_to_create:
        folder_path = project_root / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Pasta criada/verificada: {folder_path}")
    
    # Verificar e mover arquivo de dados se necessário
    data_file = "4-Carateristicas_salario.csv"
    target_location = project_root / "data" / "raw" / data_file
    
    # Locais onde o arquivo pode estar
    possible_locations = [
        project_root / data_file,                    # Raiz
        project_root / "bkp" / data_file,            # Pasta bkp
        project_root / "data" / data_file,           # Pasta data
    ]
    
    if not target_location.exists():
        for location in possible_locations:
            if location.exists():
                print(f"📁 Arquivo encontrado em: {location}")
                shutil.copy2(location, target_location)
                print(f"✅ Arquivo copiado para: {target_location}")
                break
        else:
            print(f"⚠️ Arquivo {data_file} não encontrado nos locais esperados:")
            for loc in possible_locations:
                print(f"   • {loc}")
            print(f"\n💡 Coloque o arquivo em: {target_location}")
    else:
        print(f"✅ Arquivo já está na localização correta: {target_location}")
    
    print("\n🎉 Estrutura de dados configurada!")
    print(f"📂 Dados brutos: {project_root / 'data' / 'raw'}")
    print(f"📂 Dados processados: {project_root / 'data' / 'processed'}")
    print(f"📂 Saídas: {project_root / 'output'}")

if __name__ == "__main__":
    setup_data_structure()