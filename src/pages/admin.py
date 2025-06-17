"""
Página de Administração
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import os
import shutil

def show_admin_page(data, i18n):
    """
    Mostrar página de administração
    """
    st.header("⚙️ Administração do Sistema")
    
    # Verificar se é admin (simplificado)
    if not st.session_state.get('is_admin', False):
        if st.button("🔑 Modo Admin"):
            admin_password = st.text_input("Senha Admin:", type="password")
            if admin_password == "admin123":  # Senha hardcoded para demo
                st.session_state['is_admin'] = True
                st.success("✅ Acesso admin concedido!")
                st.rerun()
            else:
                st.error("❌ Senha incorreta")
        return
    
    # Interface de admin
    admin_tab = st.selectbox(
        "Seção:",
        ["Sistema", "Dados", "Modelos", "Logs", "Configurações"]
    )
    
    if admin_tab == "Sistema":
        show_system_info(data)
    elif admin_tab == "Dados":
        show_data_management(data)
    elif admin_tab == "Modelos":
        show_model_management()
    elif admin_tab == "Logs":
        show_logs()
    elif admin_tab == "Configurações":
        show_settings()

def show_system_info(data):
    """Informações do sistema"""
    st.subheader("🖥️ Informações do Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sistema:**")
        system_info = {
            "Python": f"{os.sys.version.split()[0]}",
            "Streamlit": st.__version__,
            "Working Directory": os.getcwd(),
            "User": os.environ.get('USER', 'Unknown')
        }
        
        for key, value in system_info.items():
            st.write(f"- {key}: {value}")
    
    with col2:
        st.write("**Dados:**")
        if data is not None:
            data_info = {
                "Linhas": len(data),
                "Colunas": len(data.columns),
                "Memória (MB)": f"{data.memory_usage(deep=True).sum() / 1024**2:.2f}",
                "Valores Nulos": data.isnull().sum().sum()
            }
            
            for key, value in data_info.items():
                st.write(f"- {key}: {value}")
        else:
            st.write("- Dados não carregados")
    
    # Estrutura de diretórios
    st.subheader("📁 Estrutura do Projeto")
    
    def show_directory_tree(path, prefix="", max_depth=3, current_depth=0):
        """Mostrar árvore de diretórios"""
        if current_depth >= max_depth:
            return
        
        path = Path(path)
        if not path.exists():
            return
        
        items = sorted(path.iterdir())
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            st.text(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and not item.name.startswith('.'):
                next_prefix = prefix + ("    " if is_last else "│   ")
                show_directory_tree(item, next_prefix, max_depth, current_depth + 1)
    
    with st.expander("Ver estrutura de arquivos"):
        show_directory_tree(".")

def show_data_management(data):
    """Gerenciamento de dados"""
    st.subheader("📊 Gerenciamento de Dados")
    
    if data is None:
        st.warning("Dados não carregados")
        return
    
    # Informações dos dados
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Registros", len(data))
    
    with col2:
        st.metric("Colunas", len(data.columns))
    
    with col3:
        missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        st.metric("% Valores Ausentes", f"{missing_percentage:.1f}%")
    
    # Qualidade dos dados
    st.subheader("🔍 Qualidade dos Dados")
    
    quality_df = pd.DataFrame({
        'Coluna': data.columns,
        'Tipo': data.dtypes,
        'Não Nulos': data.count(),
        'Nulos': data.isnull().sum(),
        '% Nulos': (data.isnull().sum() / len(data) * 100).round(2)
    })
    
    st.dataframe(quality_df, use_container_width=True)
    
    # Exportar dados
    st.subheader("📤 Exportar Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Exportar CSV"):
            csv = data.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="data_export.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("💾 Exportar JSON"):
            json_data = data.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name="data_export.json",
                mime="application/json"
            )

def show_model_management():
    """Gerenciamento de modelos"""
    st.subheader("🤖 Gerenciamento de Modelos")
    
    models_dir = Path("models")
    if not models_dir.exists():
        st.warning("Diretório de modelos não existe")
        if st.button("📁 Criar diretório"):
            models_dir.mkdir()
            st.success("✅ Diretório criado!")
        return
    
    model_files = list(models_dir.glob("*.pkl"))
    
    if not model_files:
        st.info("Nenhum modelo encontrado")
        return
    
    st.write(f"**{len(model_files)} modelo(s) encontrado(s):**")
    
    for model_file in model_files:
        with st.expander(f"📊 {model_file.name}"):
            file_size = model_file.stat().st_size / 1024  # KB
            st.write(f"- Tamanho: {file_size:.2f} KB")
            st.write(f"- Modificado: {pd.Timestamp.fromtimestamp(model_file.stat().st_mtime)}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"🗑️ Deletar", key=f"delete_{model_file.name}"):
                    model_file.unlink()
                    st.success(f"✅ {model_file.name} deletado!")
                    st.rerun()
            
            with col2:
                # Download do modelo
                with open(model_file, 'rb') as f:
                    st.download_button(
                        label="📥 Download",
                        data=f.read(),
                        file_name=model_file.name,
                        mime="application/octet-stream",
                        key=f"download_{model_file.name}"
                    )

def show_logs():
    """Visualizar logs"""
    st.subheader("📋 Logs do Sistema")
    
    # Logs fictícios para demonstração
    logs = [
        {"timestamp": "2024-01-15 10:30:00", "level": "INFO", "message": "Sistema iniciado"},
        {"timestamp": "2024-01-15 10:35:00", "level": "INFO", "message": "Dados carregados com sucesso"},
        {"timestamp": "2024-01-15 10:40:00", "level": "WARNING", "message": "Valores ausentes detectados"},
        {"timestamp": "2024-01-15 11:00:00", "level": "INFO", "message": "Modelo treinado"},
        {"timestamp": "2024-01-15 11:30:00", "level": "ERROR", "message": "Erro ao salvar modelo"},
    ]
    
    logs_df = pd.DataFrame(logs)
    
    # Filtros
    col1, col2 = st.columns(2)
    
    with col1:
        log_level = st.selectbox("Nível:", ["Todos", "INFO", "WARNING", "ERROR"])
    
    with col2:
        max_logs = st.number_input("Máximo de logs:", min_value=10, max_value=1000, value=100)
    
    # Filtrar logs
    if log_level != "Todos":
        filtered_logs = logs_df[logs_df['level'] == log_level]
    else:
        filtered_logs = logs_df
    
    filtered_logs = filtered_logs.tail(max_logs)
    
    # Mostrar logs
    st.dataframe(filtered_logs, use_container_width=True)
    
    # Download logs
    if st.button("📥 Download Logs"):
        csv = filtered_logs.to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name="system_logs.csv",
            mime="text/csv"
        )

def show_settings():
    """Configurações do sistema"""
    st.subheader("⚙️ Configurações")
    
    # Configurações fictícias
    st.write("**Configurações Gerais:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        debug_mode = st.checkbox("Modo Debug", False)
        auto_refresh = st.checkbox("Auto Refresh", True)
        max_records = st.number_input("Máximo de registros:", min_value=100, max_value=10000, value=1000)
    
    with col2:
        theme = st.selectbox("Tema:", ["Claro", "Escuro", "Auto"])
        language = st.selectbox("Idioma:", ["Português", "English", "Español"])
        timezone = st.selectbox("Fuso Horário:", ["America/Sao_Paulo", "UTC", "America/New_York"])
    
    if st.button("💾 Salvar Configurações"):
        config = {
            "debug_mode": debug_mode,
            "auto_refresh": auto_refresh,
            "max_records": max_records,
            "theme": theme,
            "language": language,
            "timezone": timezone
        }
        
        # Salvar em arquivo JSON
        config_path = Path("config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        st.success("✅ Configurações salvas!")
    
    # Reset configurações
    if st.button("🔄 Reset para Padrão"):
        config_path = Path("config.json")
        if config_path.exists():
            config_path.unlink()
        st.success("✅ Configurações resetadas!")
        st.rerun()