"""Script para configurar estrutura de dados correta - VERSÃO MELHORADA"""

import os
import shutil
from pathlib import Path

def setup_data_structure():
    """Configurar estrutura de pastas e mover arquivos se necessário"""
    project_root = Path(__file__).parent
    
    print("🚀 Configurando estrutura do projeto...")
    
    # Criar estrutura de pastas COMPLETA
    folders_to_create = [
        # Dados
        "data/raw",
        "data/processed", 
        
        # Outputs
        "output/images",
        "output/analysis",
        "output/logs",
        
        # Modelos
        "models/trained",
        
        # Src structure
        "src/data",
        "src/models", 
        "src/analysis",
        "src/evaluation",
        "src/visualization",
        "src/utils",
        "src/pipelines",
        "src/config"
    ]
    
    for folder in folders_to_create:
        folder_path = project_root / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Pasta criada/verificada: {folder}")
    
    # Verificar e mover arquivo de dados se necessário
    data_file = "4-Carateristicas_salario.csv"
    target_location = project_root / "data" / "raw" / data_file
    
    # Locais onde o arquivo pode estar
    possible_locations = [
        project_root / data_file,                    # Raiz
        project_root / "bkp" / data_file,            # Pasta bkp
        project_root / "data" / data_file,           # Pasta data
        project_root / "Data" / data_file,           # Pasta Data (maiúscula)
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
            print(f"\n💡 Coloque o arquivo em qualquer uma das localizações acima")
            print(f"📍 Localização recomendada: {target_location}")
    else:
        print(f"✅ Arquivo já está na localização correta: {target_location}")
    
    # Criar arquivos __init__.py se não existirem
    init_files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/analysis/__init__.py", 
        "src/evaluation/__init__.py",
        "src/visualization/__init__.py",
        "src/utils/__init__.py",
        "src/pipelines/__init__.py",
        "src/config/__init__.py"
    ]
    
    for init_file in init_files:
        init_path = project_root / init_file
        if not init_path.exists():
            init_path.write_text('# Init file\n"""Módulo de inicialização"""\n')
            print(f"✅ Arquivo __init__.py criado: {init_file}")
    
    # Verificar se módulos críticos existem
    critical_modules = [
        "src/data/processor.py",
        "src/models/trainer.py",
        "src/visualization/plots.py",
        "src/visualization/styles.py"
    ]
    
    missing_modules = []
    for module in critical_modules:
        module_path = project_root / module
        if not module_path.exists():
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️ Módulos críticos em falta:")
        for module in missing_modules:
            print(f"   • {module}")
        print(f"\n💡 Execute o pipeline principal para criar os módulos automaticamente")
    
    print("\n🎉 Estrutura de dados configurada!")
    print(f"📂 Dados brutos: {project_root / 'data' / 'raw'}")
    print(f"📂 Dados processados: {project_root / 'data' / 'processed'}")
    print(f"📂 Saídas: {project_root / 'output'}")
    print(f"📂 Código fonte: {project_root / 'src'}")
    
    return True

def verify_structure():
    """Verificar se a estrutura está correta"""
    project_root = Path(__file__).parent
    
    required_items = [
        ("data/raw", "dir"),
        ("data/processed", "dir"),
        ("output", "dir"),
        ("src", "dir"),
        ("data/raw/4-Carateristicas_salario.csv", "file")
    ]
    
    all_good = True
    for item, item_type in required_items:
        path = project_root / item
        if item_type == "dir" and not path.is_dir():
            print(f"❌ Pasta em falta: {item}")
            all_good = False
        elif item_type == "file" and not path.is_file():
            print(f"❌ Arquivo em falta: {item}")
            all_good = False
        else:
            print(f"✅ {item}")
    
    return all_good

if __name__ == "__main__":
    print("🔧 CONFIGURAÇÃO DA ESTRUTURA DO PROJETO")
    print("=" * 50)
    
    setup_data_structure()
    
    print("\n🔍 VERIFICAÇÃO DA ESTRUTURA")
    print("=" * 50)
    
    if verify_structure():
        print("\n🎉 Estrutura está correta! Pode executar: python main.py")
    else:
        print("\n⚠️ Estrutura incompleta. Execute novamente ou configure manualmente.")