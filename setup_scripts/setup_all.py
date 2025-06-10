"""
🎯 SETUP COMPLETO DO SISTEMA - ARQUIVO ÚNICO
Sistema de Análise Salarial - Configuração Automática Total
"""

import subprocess
import sys
import os
import shutil
import json
import time
from pathlib import Path

class SystemSetup:
    """Configuração completa automática do sistema"""
    
    def __init__(self):
        # Ajustar para diretório raiz se executado da pasta setup_scripts
        if Path.cwd().name == "setup_scripts":
            os.chdir("..")
        
        self.project_root = Path.cwd()
        self.start_time = time.time()
        
    def log(self, message, level="INFO"):
        """Log com timestamp"""
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:6.1f}s] {message}")
    
    def check_python(self):
        """Verificar versão do Python"""
        self.log("🐍 Verificando Python...")
        
        if sys.version_info < (3, 8):
            self.log(f"❌ Python {sys.version} muito antigo. Requerido: 3.8+", "ERROR")
            return False
        
        self.log(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
        return True
    
    def install_dependencies(self):
        """Instalar todas as dependências"""
        self.log("📦 Instalando dependências...")
        
        dependencies = [
            'mysql-connector-python',
            'python-dotenv',
            'pandas',
            'numpy', 
            'scikit-learn',
            'matplotlib',
            'seaborn',
            'streamlit',
            'plotly',
            'joblib',
            'mlxtend',
            'xgboost',
            'lightgbm'
        ]
        
        failed = []
        for dep in dependencies:
            try:
                self.log(f"   📦 {dep}...")
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', dep, '--quiet'
                ])
            except subprocess.CalledProcessError:
                failed.append(dep)
        
        if failed:
            self.log(f"❌ Falhas: {failed}", "ERROR")
            return False
        
        self.log("✅ Dependências instaladas!")
        return True
    
    def test_imports(self):
        """Testar imports críticos"""
        self.log("🔍 Testando imports...")
        
        critical_imports = [
            'mysql.connector',
            'pandas',
            'numpy',
            'sklearn',
            'streamlit',
            'matplotlib',
            'dotenv'
        ]
        
        for module in critical_imports:
            try:
                __import__(module)
            except ImportError:
                self.log(f"❌ Import falhou: {module}", "ERROR")
                return False
        
        self.log("✅ Imports funcionando!")
        return True
    
    def create_structure(self):
        """Criar estrutura completa do projeto"""
        self.log("📁 Criando estrutura...")
        
        # Estrutura completa
        structure = {
            # Core directories
            "src": {
                "database": ["__init__.py", "connection.py", "models.py", "migration.py"],
                "pipelines": ["__init__.py", "data_pipeline.py", "ml_pipeline.py", "utils.py"],
                "utils": ["__init__.py", "logger.py", "helpers.py"],
                "visualization": ["__init__.py", "styles.py", "plots.py"],
                "config": ["__init__.py", "settings.py"]
            },
            
            # Data directories
            "data": {
                "raw": [],
                "processed": []
            },
            
            # Output directories
            "output": {
                "images": [],
                "analysis": [],
                "logs": []
            },
            
            # Models
            "models": {
                "trained": [],
                "backup": []
            },
            
            # Scripts
            "setup_scripts": [],
            
            # Backup
            "bkp": [],
            
            # Config
            "config": ["settings.py"],
            
            # Utils
            "utils": ["logging_config.py"],
            
            # Tests
            "tests": [],
            
            # Docs
            "docs": []
        }
        
        def create_recursive(base_path, structure_dict):
            for name, content in structure_dict.items():
                current_path = base_path / name
                current_path.mkdir(exist_ok=True)
                
                if isinstance(content, dict):
                    create_recursive(current_path, content)
                elif isinstance(content, list):
                    for file_name in content:
                        file_path = current_path / file_name
                        if not file_path.exists():
                            if file_name.endswith('.py'):
                                if file_name == "__init__.py":
                                    file_path.write_text('"""Módulo de inicialização"""\n')
                                else:
                                    file_path.write_text(f'"""Módulo {file_name}"""\n\n# TODO: Implementar\n')
        
        create_recursive(self.project_root, structure)
        self.log("✅ Estrutura criada!")
        return True
    
    def create_core_modules(self):
        """Criar módulos essenciais do sistema"""
        self.log("🔧 Criando módulos essenciais...")
        
        modules = {
            "src/utils/logger.py": '''"""Sistema de logging completo"""
import logging
import sys
from pathlib import Path

def setup_logging(level=logging.INFO):
    """Configurar logging detalhado"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_dir / "ml_pipeline.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Configurar logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def log_success(message):
    """Log com checkmark verde"""
    logging.info(f"✅ {message}")

def log_error(message):
    """Log de erro"""
    logging.error(f"❌ {message}")

def log_warning(message):
    """Log de aviso"""
    logging.warning(f"⚠️ {message}")
''',

            "src/visualization/styles.py": '''"""Estilos de visualização"""
import matplotlib.pyplot as plt
import seaborn as sns

def setup_matplotlib_style():
    """Configurar estilo matplotlib"""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    # Configurar seaborn
    sns.set_palette("husl")
    sns.set_context("notebook", font_scale=1.1)

def get_color_palette():
    """Obter paleta de cores consistente"""
    return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
''',

            "src/config/settings.py": '''"""Configurações centralizadas do projeto"""
import os
from pathlib import Path

# Paths do projeto
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
MODELS_DIR = BASE_DIR / "models" / "trained"
LOGS_DIR = BASE_DIR / "logs"

# Configurações de dados
DATA_CONFIG = {
    'source_file': '4-Carateristicas_salario.csv',
    'target_column': 'salary',
    'test_size': 0.2,
    'random_state': 42
}

# Configurações de modelos
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    },
    'logistic_regression': {
        'random_state': 42,
        'max_iter': 1000
    }
}

# Configurações de database
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'salary_analysis'),
    'user': os.getenv('DB_USER', 'salary_user'),
    'password': os.getenv('DB_PASSWORD', 'senha_forte')
}

# Criar diretórios se não existirem
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, 
                  IMAGES_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
''',

            "utils/logging_config.py": '''"""Configuração de logging para compatibilidade"""
import logging
import sys
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_file=None):
    """Configurar logging compatível"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger('ml_pipeline')
    logger.setLevel(log_level)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para arquivo (se especificado)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
''',

            "config/settings.py": '''"""Configurações principais do projeto"""
import os
from pathlib import Path

# Paths do projeto
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
MODELS_DIR = BASE_DIR / "models" / "trained"
LOGS_DIR = OUTPUT_DIR / "logs"

# Configurações de dados
DATA_CONFIG = {
    'source_file': '4-Carateristicas_salario.csv',
    'target_column': 'salary',
    'test_size': 0.2,
    'random_state': 42
}

# Configurações de modelos
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    },
    'logistic_regression': {
        'random_state': 42,
        'max_iter': 1000
    }
}

# Validação de ranges
VALIDATION_RANGES = {
    'age': (17, 100),
    'education-num': (1, 16),
    'hours-per-week': (1, 99)
}

# Criar diretórios se não existirem
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, 
                  IMAGES_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
'''
        }
        
        for file_path, content in modules.items():
            path = self.project_root / file_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(content)
        
        self.log("✅ Módulos essenciais criados!")
        return True
    
    def setup_data_files(self):
        """Configurar arquivos de dados"""
        self.log("📄 Configurando arquivos de dados...")
        
        # Procurar arquivo CSV principal
        csv_file = "4-Carateristicas_salario.csv"
        target_location = self.project_root / "data" / "raw" / csv_file
        
        # Locais possíveis
        possible_locations = [
            self.project_root / csv_file,
            self.project_root / "bkp" / csv_file,
            self.project_root / "data" / csv_file,
            self.project_root / "Data" / csv_file,
        ]
        
        if not target_location.exists():
            for location in possible_locations:
                if location.exists():
                    self.log(f"   📁 Encontrado: {location}")
                    shutil.copy2(location, target_location)
                    self.log(f"   ✅ Copiado para: {target_location}")
                    break
            else:
                self.log(f"   ⚠️ {csv_file} não encontrado", "WARNING")
                self.log(f"   💡 Coloque em: {target_location}")
        else:
            self.log(f"   ✅ CSV já está em: {target_location}")
        
        return True
    
    def check_mysql(self):
        """Verificar MySQL"""
        self.log("🗄️ Verificando MySQL...")
        
        try:
            result = subprocess.run(['mysql', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.log("   ✅ MySQL instalado")
                
                # Verificar se está rodando
                try:
                    subprocess.run(['pgrep', 'mysql'], 
                                  capture_output=True, check=True)
                    self.log("   ✅ MySQL rodando")
                    return True
                except subprocess.CalledProcessError:
                    self.log("   ⚠️ MySQL parado. Inicie: brew services start mysql", "WARNING")
                    return False
            else:
                self.log("   ❌ MySQL não encontrado", "WARNING")
                return False
        except FileNotFoundError:
            self.log("   ❌ MySQL não instalado", "WARNING")
            self.log("   💡 Instale: brew install mysql (macOS)")
            return False
    
    def create_env_file(self):
        """Criar arquivo .env"""
        self.log("📄 Criando .env...")
        
        env_content = f"""# =================================================
# Sistema de Análise Salarial - Configuração Automática
# Gerado em: {time.strftime('%Y-%m-%d %H:%M:%S')}
# =================================================

# Banco de Dados MySQL (OBRIGATÓRIO)
DB_HOST=localhost
DB_NAME=salary_analysis
DB_USER=salary_user
DB_PASSWORD=senha_forte

# Configurações do Sistema
USE_DATABASE=true
AUTO_MIGRATE=true

# Configurações de Log
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Configurações de ML
RANDOM_STATE=42
TEST_SIZE=0.2

# Configurações de Performance
CACHE_TTL=300
MAX_WORKERS=4

# Configurações de Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true

# =================================================
# EDITE AS CREDENCIAIS DO MYSQL CONFORME NECESSÁRIO
# =================================================
"""
        
        env_path = self.project_root / ".env"
        if not env_path.exists():
            env_path.write_text(env_content)
            self.log("   ✅ .env criado")
        else:
            self.log("   ✅ .env já existe")
        
        return True
    
    def test_mysql_connection(self):
        """Testar conexão MySQL"""
        self.log("🔍 Testando MySQL...")
        
        try:
            import mysql.connector
            
            # Configurações de teste
            test_configs = [
                {'user': 'root', 'password': ''},
                {'user': 'root', 'password': 'root'},
                {'user': 'root', 'password': 'password'},
                {'user': 'salary_user', 'password': 'senha_forte'}
            ]
            
            for config in test_configs:
                try:
                    config['host'] = 'localhost'
                    conn = mysql.connector.connect(**config, connection_timeout=3)
                    if conn.is_connected():
                        self.log(f"   ✅ Conexão OK: {config['user']}")
                        conn.close()
                        return config
                except mysql.connector.Error:
                    continue
            
            self.log("   ⚠️ Nenhuma conexão funcionou", "WARNING")
            return None
            
        except ImportError:
            self.log("   ❌ mysql-connector não disponível", "ERROR")
            return None
    
    def setup_mysql_database(self, working_config):
        """Configurar database MySQL"""
        if not working_config:
            self.log("   ⚠️ Pulando configuração MySQL", "WARNING")
            return False
        
        self.log("🗄️ Configurando database...")
        
        try:
            import mysql.connector
            
            conn = mysql.connector.connect(**working_config)
            cursor = conn.cursor()
            
            # Criar database
            cursor.execute("CREATE DATABASE IF NOT EXISTS salary_analysis CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            
            # Criar usuário se conectado como root
            if working_config['user'] == 'root':
                try:
                    cursor.execute("CREATE USER IF NOT EXISTS 'salary_user'@'localhost' IDENTIFIED BY 'senha_forte'")
                    cursor.execute("GRANT ALL PRIVILEGES ON salary_analysis.* TO 'salary_user'@'localhost'")
                    cursor.execute("FLUSH PRIVILEGES")
                    self.log("   ✅ Usuário salary_user criado")
                except mysql.connector.Error:
                    self.log("   ⚠️ Usuário já existe", "WARNING")
            
            cursor.close()
            conn.close()
            self.log("   ✅ Database configurada!")
            return True
            
        except Exception as e:
            self.log(f"   ❌ Erro: {e}", "ERROR")
            return False
    
    def create_test_files(self):
        """Criar arquivos de teste"""
        self.log("🧪 Criando testes...")
        
        # Test system
        test_system_content = '''#!/usr/bin/env python3
"""Teste completo do sistema"""

import os
import sys
from pathlib import Path

def test_system():
    """Testar todos os componentes"""
    # Ajustar path se necessário
    if Path.cwd().name == "setup_scripts":
        os.chdir("..")
    
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
    
    # Teste 2: Estrutura
    required_dirs = ["src/database", "src/pipelines", "data", "output"]
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory}")
    
    # Teste 3: Arquivos importantes
    required_files = [".env", "main.py", "app.py"]
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
    
    # Teste 4: MySQL (se disponível)
    try:
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
        
        conn = mysql.connector.connect(**config, connection_timeout=3)
        if conn.is_connected():
            print("✅ Conexão MySQL")
            conn.close()
    except Exception as e:
        print(f"⚠️  MySQL: {e}")
    
    print("\\n🎯 Teste concluído!")
    return True

if __name__ == "__main__":
    test_system()
'''
        
        # Diagnóstico rápido
        diagnose_content = '''#!/usr/bin/env python3
"""Diagnóstico rápido do sistema"""

def diagnose():
    print("🔍 DIAGNÓSTICO RÁPIDO")
    print("=" * 30)
    
    # Python
    import sys
    print(f"🐍 Python: {sys.version_info.major}.{sys.version_info.minor}")
    
    # Dependências
    deps = ['mysql.connector', 'pandas', 'numpy', 'sklearn', 'streamlit']
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep}")
    
    # Estrutura
    from pathlib import Path
    dirs = ['src/', 'data/', 'output/', 'models/']
    for d in dirs:
        print(f"✅ {d}" if Path(d).exists() else f"❌ {d}")
    
    # Arquivos
    files = ['.env', 'main.py', 'app.py']
    for f in files:
        print(f"✅ {f}" if Path(f).exists() else f"❌ {f}")

if __name__ == "__main__":
    diagnose()
'''
        
        # Salvar arquivos
        test_files = {
            "setup_scripts/test_system.py": test_system_content,
            "diagnose.py": diagnose_content
        }
        
        for file_path, content in test_files.items():
            path = self.project_root / file_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            path.chmod(0o755)
        
        self.log("✅ Arquivos de teste criados!")
        return True
    
    def create_requirements(self):
        """Criar requirements.txt completo"""
        self.log("📋 Criando requirements.txt...")
        
        requirements_content = """# Sistema de Análise Salarial - Dependências
# Core ML
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Visualização
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Dashboard
streamlit>=1.25.0
streamlit-option-menu>=0.3.6

# Database
mysql-connector-python>=8.1.0
python-dotenv>=1.0.0

# Machine Learning Avançado
xgboost>=1.7.0
lightgbm>=3.3.0
joblib>=1.3.0

# Regras de Associação
mlxtend>=0.22.0

# Utilitários
pathlib2>=2.3.7
tqdm>=4.65.0

# Análise de dados
openpyxl>=3.1.0
xlsxwriter>=3.1.0

# Interpretabilidade (opcional)
shap>=0.42.0

# Monitoramento (opcional)
psutil>=5.9.0
"""
        
        req_path = self.project_root / "requirements.txt"
        req_path.write_text(requirements_content)
        self.log("✅ requirements.txt criado!")
        return True
    
    def create_documentation(self):
        """Criar documentação básica"""
        self.log("📚 Criando documentação...")
        
        readme_content = f"""# 📊 Sistema de Análise Salarial

**Configuração automática concluída em: {time.strftime('%Y-%m-%d %H:%M:%S')}**

## 🚀 Início Rápido

```bash
# 1. Testar sistema
python diagnose.py

# 2. Configurar banco (se necessário)
python main.py --setup-db

# 3. Executar pipeline
python main.py

# 4. Abrir dashboard
streamlit run app.py
```

## 📁 Estrutura do Projeto

```
📁 projeto_final/
├── 📄 main.py                    # Pipeline principal
├── 📄 app.py                     # Dashboard Streamlit
├── 📄 .env                       # Configurações
├── 📄 requirements.txt           # Dependências
├── 📄 diagnose.py               # Diagnóstico rápido
├── 📁 src/                      # Código fonte
│   ├── 📁 database/             # Módulos de banco
│   ├── 📁 pipelines/            # Pipelines de dados e ML
│   ├── 📁 utils/                # Utilitários
│   └── 📁 visualization/        # Visualizações
├── 📁 data/                     # Dados
│   ├── 📁 raw/                  # Dados brutos
│   └── 📁 processed/            # Dados processados
├── 📁 output/                   # Saídas
│   ├── 📁 images/               # Gráficos
│   └── 📁 analysis/             # Análises
├── 📁 models/                   # Modelos ML
├── 📁 setup_scripts/            # Scripts de configuração
└── 📁 logs/                     # Logs do sistema
```

## 🔧 Configuração

### MySQL
1. Edite `.env` com suas credenciais MySQL
2. Execute: `python main.py --setup-db`

### Dados
1. Coloque `4-Carateristicas_salario.csv` em `data/raw/`
2. Execute: `python main.py --migrate`

## 🧪 Teste

```bash
# Teste completo
python setup_scripts/test_system.py

# Diagnóstico rápido
python diagnose.py
```

## 📊 Uso

```bash
# Pipeline completo
python main.py

# Dashboard
streamlit run app.py
```

## 🔧 Resolução de Problemas

### MySQL
```bash
# Verificar serviço
brew services list | grep mysql

# Iniciar
brew services start mysql

# Conectar
mysql -u salary_user -p salary_analysis
```

### Dependências
```bash
# Reinstalar
pip install -r requirements.txt

# Verificar
python diagnose.py
```

## 📋 Comandos Úteis

```bash
# Logs
tail -f logs/app.log

# Backup MySQL
mysqldump -u salary_user -p salary_analysis > backup.sql

# Restaurar
mysql -u salary_user -p salary_analysis < backup.sql
```

---
**Sistema configurado automaticamente pelo setup_all.py**
"""
        
        readme_path = self.project_root / "README.md"
        readme_path.write_text(readme_content)
        self.log("✅ README.md criado!")
        return True
    
    def run_setup(self):
        """Executar setup completo"""
        self.log("🎯 INÍCIO DO SETUP COMPLETO")
        self.log("="*50)
        
        steps = [
            ("Python", self.check_python),
            ("Dependências", self.install_dependencies),
            ("Imports", self.test_imports),
            ("Estrutura", self.create_structure),
            ("Módulos", self.create_core_modules),
            ("Dados", self.setup_data_files),
            ("MySQL", self.check_mysql),
            ("Ambiente", self.create_env_file),
            ("Database", lambda: self.setup_mysql_database(self.test_mysql_connection())),
            ("Testes", self.create_test_files),
            ("Requirements", self.create_requirements),
            ("Documentação", self.create_documentation)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            try:
                if not step_func():
                    failed_steps.append(step_name)
            except Exception as e:
                self.log(f"❌ Erro em {step_name}: {e}", "ERROR")
                failed_steps.append(step_name)
        
        # Relatório final
        elapsed = time.time() - self.start_time
        self.log("="*50)
        
        if failed_steps:
            self.log(f"⚠️ Setup concluído com avisos em {elapsed:.1f}s", "WARNING")
            self.log(f"Passos com problemas: {failed_steps}")
        else:
            self.log(f"🎉 Setup completo concluído em {elapsed:.1f}s!")
        
        self.show_next_steps(failed_steps)
        
        return len(failed_steps) == 0
    
    def show_next_steps(self, failed_steps):
        """Mostrar próximos passos"""
        self.log("\n🚀 PRÓXIMOS PASSOS:")
        self.log("-" * 30)
        
        if failed_steps:
            self.log("1. 🔧 Resolver problemas:")
            for step in failed_steps:
                self.log(f"   • {step}")
        
        self.log("2. 🧪 Testar sistema:")
        self.log("   python diagnose.py")
        
        self.log("3. 📊 Configurar dados:")
        self.log("   • Coloque CSV em data/raw/")
        self.log("   • Configure MySQL no .env")
        
        self.log("4. 🚀 Executar pipeline:")
        self.log("   python main.py")
        
        self.log("5. 🌐 Abrir dashboard:")
        self.log("   streamlit run app.py")
        
        self.log("\n📋 ARQUIVOS IMPORTANTES:")
        self.log("• .env               → Configurações")
        self.log("• diagnose.py        → Teste rápido")
        self.log("• README.md          → Documentação") 
        self.log("• requirements.txt   → Dependências")

def main():
    """Função principal"""
    setup = SystemSetup()
    
    try:
        success = setup.run_setup()
        
        if success:
            print("\n" + "="*60)
            print("🎉 SISTEMA CONFIGURADO COM SUCESSO!")
            print("="*60)
            print("\n💡 Execute: python diagnose.py")
        else:
            print("\n" + "="*60) 
            print("⚠️ SETUP CONCLUÍDO COM AVISOS")
            print("="*60)
            print("\n💡 Verifique os problemas acima")
            
    except KeyboardInterrupt:
        print("\n⚠️ Setup interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro crítico no setup: {e}")

if __name__ == "__main__":
    main()