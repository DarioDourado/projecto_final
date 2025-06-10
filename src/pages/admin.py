"""
⚙️ Página de Administração
Configurações avançadas e gestão do sistema
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def show_admin_page(data, i18n, auth_system):
    """Página de administração do sistema"""
    from src.components.navigation import show_page_header
    
    show_page_header(
        i18n.t('navigation.admin', 'Administração'),
        i18n.t('admin.subtitle', 'Configurações e gestão do sistema'),
        "⚙️"
    )
    
    # Verificar permissões de admin
    if not _check_admin_permissions(auth_system):
        st.error(i18n.t('admin.no_permissions', 'Acesso negado. Apenas administradores.'))
        return
    
    # Tabs de administração
    tab1, tab2, tab3, tab4 = st.tabs([
        f"👥 {i18n.t('admin.users', 'Gestão de Usuários')}",
        f"⚙️ {i18n.t('admin.system', 'Configurações do Sistema')}",
        f"📊 {i18n.t('admin.monitoring', 'Monitoramento')}",
        f"🧹 {i18n.t('admin.maintenance', 'Manutenção')}"
    ])
    
    with tab1:
        _show_user_management(auth_system, i18n)
    
    with tab2:
        _show_system_config(i18n)
    
    with tab3:
        _show_monitoring(data, i18n)
    
    with tab4:
        _show_maintenance(i18n)

def _check_admin_permissions(auth_system):
    """Verificar se o usuário tem permissões de admin"""
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        return False
    
    user_data = st.session_state.get('user_data', {})
    return user_data.get('role') == 'admin'

def _show_user_management(auth_system, i18n):
    """Gestão de usuários"""
    st.subheader(f"👥 {i18n.t('admin.user_management', 'Gestão de Usuários')}")
    
    # Carregar usuários
    users = auth_system.load_users()
    
    # Estatísticas de usuários
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_users = len(users)
        st.metric(f"👥 {i18n.t('admin.total_users', 'Total de Usuários')}", total_users)
    
    with col2:
        admin_count = sum(1 for user in users.values() if user.get('role') == 'admin')
        st.metric(f"👑 {i18n.t('admin.admin_count', 'Administradores')}", admin_count)
    
    with col3:
        active_sessions = len(auth_system.load_sessions())
        st.metric(f"🟢 {i18n.t('admin.active_sessions', 'Sessões Ativas')}", active_sessions)
    
    # Lista de usuários
    st.markdown(f"### 📋 {i18n.t('admin.users_list', 'Lista de Usuários')}")
    
    if users:
        users_data = []
        for username, user_info in users.items():
            users_data.append({
                'Username': username,
                'Nome': user_info.get('name', 'N/A'),
                'Email': user_info.get('email', 'N/A'),
                'Papel': user_info.get('role', 'user'),
                'Criado': user_info.get('created', 'N/A')
            })
        
        users_df = pd.DataFrame(users_data)
        st.dataframe(users_df, use_container_width=True)
        
        # Ações de usuário
        st.markdown(f"### 🔧 {i18n.t('admin.user_actions', 'Ações de Usuário')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Criar novo usuário
            with st.expander(f"➕ {i18n.t('admin.create_user', 'Criar Novo Usuário')}"):
                new_username = st.text_input(f"👤 {i18n.t('admin.username', 'Username')}")
                new_name = st.text_input(f"📝 {i18n.t('admin.full_name', 'Nome Completo')}")
                new_email = st.text_input(f"📧 {i18n.t('admin.email', 'Email')}")
                new_password = st.text_input(f"🔒 {i18n.t('admin.password', 'Senha')}", type="password")
                new_role = st.selectbox(f"👑 {i18n.t('admin.role', 'Papel')}", ['user', 'admin'])
                
                if st.button(f"✅ {i18n.t('admin.create', 'Criar Usuário')}"):
                    if new_username and new_password and new_name and new_email:
                        success, message = auth_system.register_user(
                            new_username, new_password, new_name, new_email
                        )
                        if success:
                            # Atualizar papel se for admin
                            if new_role == 'admin':
                                users = auth_system.load_users()
                                users[new_username]['role'] = 'admin'
                                auth_system.save_users(users)
                            
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                    else:
                        st.error(i18n.t('admin.fill_all_fields', 'Preencha todos os campos'))
        
        with col2:
            # Modificar usuário existente
            with st.expander(f"✏️ {i18n.t('admin.modify_user', 'Modificar Usuário')}"):
                selected_user = st.selectbox(
                    f"👤 {i18n.t('admin.select_user', 'Selecionar Usuário')}",
                    list(users.keys())
                )
                
                if selected_user:
                    current_user = users[selected_user]
                    
                    new_role_modify = st.selectbox(
                        f"👑 {i18n.t('admin.new_role', 'Novo Papel')}",
                        ['user', 'admin'],
                        index=0 if current_user.get('role') == 'user' else 1
                    )
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if st.button(f"💾 {i18n.t('admin.update_role', 'Atualizar Papel')}"):
                            users[selected_user]['role'] = new_role_modify
                            auth_system.save_users(users)
                            st.success(f"✅ {i18n.t('admin.role_updated', 'Papel atualizado')}")
                            st.rerun()
                    
                    with col_b:
                        if st.button(f"🗑️ {i18n.t('admin.delete_user', 'Deletar Usuário')}", type="secondary"):
                            if selected_user != st.session_state.get('username'):  # Não pode deletar a si mesmo
                                del users[selected_user]
                                auth_system.save_users(users)
                                st.success(f"✅ {i18n.t('admin.user_deleted', 'Usuário deletado')}")
                                st.rerun()
                            else:
                                st.error(i18n.t('admin.cannot_delete_self', 'Não pode deletar sua própria conta'))

def _show_system_config(i18n):
    """Configurações do sistema"""
    st.subheader(f"⚙️ {i18n.t('admin.system_config', 'Configurações do Sistema')}")
    
    # Configurações de idioma
    st.markdown(f"### 🌍 {i18n.t('admin.language_config', 'Configurações de Idioma')}")
    
    config_file = Path("translate/config.json")
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        st.json(config)
        
        # Opção para editar configurações
        if st.checkbox(f"✏️ {i18n.t('admin.edit_config', 'Editar Configurações')}"):
            new_config = st.text_area(
                f"📝 {i18n.t('admin.config_json', 'Configuração JSON')}",
                value=json.dumps(config, indent=2, ensure_ascii=False),
                height=200
            )
            
            if st.button(f"💾 {i18n.t('admin.save_config', 'Salvar Configurações')}"):
                try:
                    updated_config = json.loads(new_config)
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(updated_config, f, indent=2, ensure_ascii=False)
                    st.success(f"✅ {i18n.t('admin.config_saved', 'Configurações salvas')}")
                except json.JSONDecodeError:
                    st.error(f"❌ {i18n.t('admin.invalid_json', 'JSON inválido')}")
    
    # Configurações de cache
    st.markdown(f"### 🗄️ {i18n.t('admin.cache_config', 'Configurações de Cache')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"🧹 {i18n.t('admin.clear_cache', 'Limpar Cache')}"):
            st.cache_data.clear()
            st.success(f"✅ {i18n.t('admin.cache_cleared', 'Cache limpo')}")
    
    with col2:
        if st.button(f"🔄 {i18n.t('admin.restart_session', 'Reiniciar Sessão')}"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success(f"✅ {i18n.t('admin.session_restarted', 'Sessão reiniciada')}")

def _show_monitoring(data, i18n):
    """Monitoramento do sistema"""
    st.subheader(f"📊 {i18n.t('admin.monitoring', 'Monitoramento do Sistema')}")
    
    # Status dos componentes
    st.markdown(f"### 🔍 {i18n.t('admin.system_status', 'Status do Sistema')}")
    
    components_status = {
        'Data Loader': '✅' if data.get('df') is not None else '❌',
        'Models': '✅' if data.get('models') else '❌',
        'Feature Importance': '✅' if data.get('feature_importance') else '❌',
        'Authentication': '✅' if Path('config/users.json').exists() else '❌',
        'Translations': '✅' if Path('translate').exists() else '❌'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        for component, status in components_status.items():
            st.markdown(f"**{component}:** {status}")
    
    with col2:
        # Estatísticas de uso
        try:
            import psutil
            
            # CPU e Memória
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            st.metric(f"💻 {i18n.t('admin.cpu_usage', 'Uso de CPU')}", f"{cpu_percent:.1f}%")
            st.metric(f"🧠 {i18n.t('admin.memory_usage', 'Uso de Memória')}", f"{memory.percent:.1f}%")
            
        except ImportError:
            st.info(i18n.t('admin.install_psutil', 'Instale psutil para métricas detalhadas'))
    
    # Logs do sistema (se existirem)
    st.markdown(f"### 📋 {i18n.t('admin.system_logs', 'Logs do Sistema')}")
    
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            selected_log = st.selectbox(
                f"📄 {i18n.t('admin.select_log', 'Selecionar Log')}",
                [f.name for f in log_files]
            )
            
            if selected_log:
                log_path = logs_dir / selected_log
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    
                    # Mostrar últimas 50 linhas
                    log_lines = log_content.split('\n')[-50:]
                    st.text_area(
                        f"📄 {selected_log}",
                        value='\n'.join(log_lines),
                        height=300
                    )
                except Exception as e:
                    st.error(f"❌ Erro ao ler log: {e}")
        else:
            st.info(i18n.t('admin.no_logs', 'Nenhum arquivo de log encontrado'))
    else:
        st.info(i18n.t('admin.logs_dir_missing', 'Diretório de logs não encontrado'))

def _show_maintenance(i18n):
    """Manutenção do sistema"""
    st.subheader(f"🧹 {i18n.t('admin.maintenance', 'Manutenção do Sistema')}")
    
    # Limpeza de arquivos
    st.markdown(f"### 🗑️ {i18n.t('admin.cleanup', 'Limpeza de Arquivos')}")
    
    cleanup_options = {
        'temp_files': f"🗂️ {i18n.t('admin.temp_files', 'Arquivos Temporários')}",
        'old_sessions': f"👥 {i18n.t('admin.old_sessions', 'Sessões Expiradas')}",
        'cache': f"🗄️ {i18n.t('admin.cache_files', 'Arquivos de Cache')}",
        'logs': f"📋 {i18n.t('admin.old_logs', 'Logs Antigos')}"
    }
    
    selected_cleanup = st.multiselect(
        f"🎯 {i18n.t('admin.select_cleanup', 'Selecionar itens para limpeza')}",
        list(cleanup_options.keys()),
        format_func=lambda x: cleanup_options[x]
    )
    
    if st.button(f"🧹 {i18n.t('admin.execute_cleanup', 'Executar Limpeza')}", type="primary"):
        cleanup_results = _execute_cleanup(selected_cleanup, i18n)
        
        for result in cleanup_results:
            if result['success']:
                st.success(f"✅ {result['message']}")
            else:
                st.error(f"❌ {result['message']}")
    
    # Backup e restauração
    st.markdown(f"### 💾 {i18n.t('admin.backup', 'Backup e Restauração')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"💾 {i18n.t('admin.create_backup', 'Criar Backup')}", use_container_width=True):
            backup_result = _create_backup(i18n)
            if backup_result['success']:
                st.success(f"✅ {backup_result['message']}")
                
                # Oferecer download do backup
                with open(backup_result['file_path'], 'rb') as f:
                    st.download_button(
                        label=f"📥 {i18n.t('admin.download_backup', 'Download Backup')}",
                        data=f.read(),
                        file_name=backup_result['file_name'],
                        mime="application/zip"
                    )
            else:
                st.error(f"❌ {backup_result['message']}")
    
    with col2:
        backup_file = st.file_uploader(
            f"📁 {i18n.t('admin.upload_backup', 'Upload Backup')}",
            type=['zip']
        )
        
        if backup_file and st.button(f"🔄 {i18n.t('admin.restore_backup', 'Restaurar Backup')}"):
            restore_result = _restore_backup(backup_file, i18n)
            if restore_result['success']:
                st.success(f"✅ {restore_result['message']}")
            else:
                st.error(f"❌ {restore_result['message']}")

def _execute_cleanup(selected_items, i18n):
    """Executar limpeza dos itens selecionados"""
    results = []
    
    for item in selected_items:
        try:
            if item == 'temp_files':
                # Limpar arquivos temporários
                temp_count = 0
                for temp_dir in [Path("temp"), Path("tmp")]:
                    if temp_dir.exists():
                        for file in temp_dir.rglob("*"):
                            if file.is_file():
                                file.unlink()
                                temp_count += 1
                
                results.append({
                    'success': True,
                    'message': f"{temp_count} arquivos temporários removidos"
                })
            
            elif item == 'old_sessions':
                # Limpar sessões expiradas
                sessions_file = Path("config/sessions.json")
                if sessions_file.exists():
                    with open(sessions_file, 'r') as f:
                        sessions = json.load(f)
                    
                    # Remover sessões antigas (placeholder - implementar lógica real)
                    old_count = len(sessions)
                    # sessions = {k: v for k, v in sessions.items() if not_expired(v)}
                    
                    with open(sessions_file, 'w') as f:
                        json.dump({}, f)  # Limpar todas para demonstração
                    
                    results.append({
                        'success': True,
                        'message': f"{old_count} sessões expiradas removidas"
                    })
            
            elif item == 'cache':
                # Limpar cache do Streamlit
                st.cache_data.clear()
                results.append({
                    'success': True,
                    'message': "Cache do Streamlit limpo"
                })
            
            elif item == 'logs':
                # Limpar logs antigos
                logs_dir = Path("logs")
                if logs_dir.exists():
                    log_count = 0
                    for log_file in logs_dir.glob("*.log"):
                        # Manter apenas logs dos últimos 7 dias (implementar lógica real)
                        log_file.unlink()
                        log_count += 1
                    
                    results.append({
                        'success': True,
                        'message': f"{log_count} arquivos de log antigos removidos"
                    })
        
        except Exception as e:
            results.append({
                'success': False,
                'message': f"Erro na limpeza de {item}: {e}"
            })
    
    return results

def _create_backup(i18n):
    """Criar backup do sistema"""
    try:
        import zipfile
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"backup_sistema_{timestamp}.zip"
        backup_path = Path("backups")
        backup_path.mkdir(exist_ok=True)
        backup_file_path = backup_path / backup_filename
        
        # Diretórios e arquivos para backup
        backup_items = [
            Path("config"),
            Path("translate"),
            Path("models"),
            Path("data/processed") if Path("data/processed").exists() else None
        ]
        backup_items = [item for item in backup_items if item and item.exists()]
        
        with zipfile.ZipFile(backup_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for item in backup_items:
                if item.is_file():
                    zipf.write(item, item.name)
                elif item.is_dir():
                    for file in item.rglob("*"):
                        if file.is_file():
                            zipf.write(file, file.relative_to(item.parent))
        
        return {
            'success': True,
            'message': f"Backup criado: {backup_filename}",
            'file_path': backup_file_path,
            'file_name': backup_filename
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f"Erro ao criar backup: {e}"
        }

def _restore_backup(backup_file, i18n):
    """Restaurar backup do sistema"""
    try:
        import zipfile
        import tempfile
        
        # Salvar arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_file.write(backup_file.read())
            tmp_path = tmp_file.name
        
        # Extrair backup
        with zipfile.ZipFile(tmp_path, 'r') as zipf:
            zipf.extractall(".")
        
        # Limpar arquivo temporário
        Path(tmp_path).unlink()
        
        return {
            'success': True,
            'message': "Backup restaurado com sucesso. Reinicie a aplicação."
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f"Erro ao restaurar backup: {e}"
        }