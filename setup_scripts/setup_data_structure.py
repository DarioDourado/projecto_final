"""
Script para configurar estrutura de dados do projeto
"""

import shutil
from pathlib import Path
import os

def setup_data_structure():
    """Configurar estrutura de dados necessária"""
    print("🔧 Configurando estrutura de dados...")
    
    # Criar diretórios necessários
    directories = [
        "data/raw",
        "data/processed", 
        "output/analysis",
        "output/models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Diretório criado: {directory}")
    
    # Procurar arquivo CSV em locais possíveis
    csv_name = "4-Carateristicas_salario.csv"
    possible_locations = [
        Path(csv_name),
        Path(f"bkp/{csv_name}"),
        Path(f"data/{csv_name}"),
        Path(f"src/data/{csv_name}"),
        Path(f"../{csv_name}"),
        Path(f"../../{csv_name}")
    ]
    
    target_location = Path("data/raw/4-Carateristicas_salario.csv")
    
    # Verificar se já existe no local correto
    if target_location.exists():
        print(f"✅ Arquivo já existe: {target_location}")
        return True
    
    # Procurar arquivo em outros locais
    csv_found = False
    for location in possible_locations:
        if location.exists():
            print(f"📁 Arquivo encontrado: {location}")
            try:
                shutil.copy2(location, target_location)
                print(f"✅ Arquivo copiado para: {target_location}")
                csv_found = True
                break
            except Exception as e:
                print(f"❌ Erro ao copiar: {e}")
                continue
    
    if not csv_found:
        print("❌ Arquivo CSV não encontrado!")
        print("💡 Soluções:")
        print("   1. Coloque '4-Carateristicas_salario.csv' na pasta 'data/raw/'")
        print("   2. Ou coloque na raiz do projeto")
        print("   3. Ou configure o banco de dados com: python main.py --setup-db")
        return False
    
    return True

def check_file_permissions():
    """Verificar permissões de arquivos"""
    csv_path = Path("data/raw/4-Carateristicas_salario.csv")
    
    if csv_path.exists():
        try:
            # Testar leitura
            with open(csv_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            
            print(f"✅ Arquivo legível: {len(first_line)} caracteres na primeira linha")
            print(f"📊 Tamanho: {csv_path.stat().st_size / 1024 / 1024:.1f} MB")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao ler arquivo: {e}")
            return False
    
    return False

if __name__ == "__main__":
    print("🚀 Configurando projeto...")
    
    # Configurar estrutura
    setup_success = setup_data_structure()
    
    if setup_success:
        # Verificar permissões
        permissions_ok = check_file_permissions()
        
        if permissions_ok:
            print("\n✅ CONFIGURAÇÃO CONCLUÍDA!")
            print("🔄 Execute novamente: python main.py")
        else:
            print("\n⚠️ Estrutura criada mas arquivo com problemas")
    else:
        print("\n❌ Configuração falhou")
        print("📋 Verifique se o arquivo '4-Carateristicas_salario.csv' existe")