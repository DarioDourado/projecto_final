"""
🧭 Componente de Navegação
Sistema de navegação e informações do usuário
"""

import streamlit as st
import time

def show_user_info(auth_system, i18n):
    """Mostrar informações do usuário logado na sidebar"""
    if 'user_data' in st.session_state:
        user_data = st.session_state.user_data
        username = st.session_state.get('username', 'Unknown')
        user_name = user_data.get('name', username)
        user_role = user_data.get('role', 'guest')
        user_email = user_data.get('email', 'N/A')
        
        with st.sidebar:
            st.markdown("---")
            
            # Card de informações do usuário
            st.markdown(f"""
            <div class="user-info-card">
                <h4>👤 {i18n.t('auth.logged_user', 'Usuário Logado')}</h4>
                <p><strong>Nome:</strong> {user_name}</p>
                <p><strong>{i18n.t('auth.role', 'Papel')}:</strong> {user_role.title()}</p>
                <p><strong>Email:</strong> {user_email}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Botão de logout
            if st.button(f"🚪 {i18n.t('auth.logout', 'Logout')}", use_container_width=True, type="secondary"):
                handle_logout(auth_system, i18n)

def handle_logout(auth_system, i18n):
    """Processar logout do usuário"""
    try:
        # Fazer logout no sistema de autenticação
        if 'session_id' in st.session_state:
            auth_system.logout(st.session_state.session_id)
        
        # Limpar estado da sessão
        session_keys = ['authenticated', 'session_id', 'user_data', 'username', 'current_page', 'filters']
        for key in session_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        # Mostrar mensagem de sucesso
        st.success(i18n.t('auth.logout_success', 'Logout realizado com sucesso!'))
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"Erro no logout: {e}")

def show_breadcrumbs(pages, i18n):
    """Mostrar breadcrumbs de navegação"""
    if not pages:
        return
    
    breadcrumb_items = []
    for i, (page_name, page_key) in enumerate(pages):
        if i == len(pages) - 1:  # Página atual
            breadcrumb_items.append(f"<span style='font-weight: bold;'>{page_name}</span>")
        else:
            breadcrumb_items.append(f"<a href='#' style='text-decoration: none; color: #667eea;'>{page_name}</a>")
    
    breadcrumb_html = " > ".join(breadcrumb_items)
    
    st.markdown(f"""
    <div style="padding: 0.5rem 0; color: #6c757d; font-size: 0.9rem;">
        🏠 {breadcrumb_html}
    </div>
    """, unsafe_allow_html=True)

def show_page_header(title, subtitle=None, icon="📊"):
    """Mostrar header da página"""
    st.markdown(f"""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #2c3e50; margin-bottom: 0.5rem;">
            {icon} {title}
        </h1>
        {f'<p style="color: #6c757d; font-size: 1.1rem; margin: 0;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def create_navigation_menu(pages, current_page, i18n):
    """Criar menu de navegação lateral"""
    st.markdown(f"### 🧭 {i18n.t('navigation.title', 'Navegação')}")
    
    for page_key, icon, roles in pages:
        page_name = i18n.t(page_key, page_key.split('.')[-1])
        
        # Verificar se é a página atual
        is_current = current_page == page_name
        
        # Estilo do botão baseado no status
        button_type = "primary" if is_current else "secondary"
        
        if st.button(f"{icon} {page_name}", 
                    use_container_width=True, 
                    type=button_type,
                    disabled=is_current):
            return page_name
    
    return None

def show_filters_sidebar(df, i18n):
    """Mostrar filtros na sidebar"""
    if df is None or len(df) == 0:
        return {}
    
    st.markdown(f"### 🔍 {i18n.t('filters.title', 'Filtros')}")
    
    filters = {}
    
    # Botão para limpar filtros
    if st.button(f"🗑️ {i18n.t('filters.clear', 'Limpar Filtros')}", use_container_width=True):
        return {}
    
    # Filtro de salário
    if 'salary' in df.columns:
        salary_options = [str(x) for x in df['salary'].unique() if str(x) != 'Unknown' and pd.notna(x)]
        if salary_options:
            selected_salaries = st.multiselect(
                f"💰 {i18n.t('data.salary', 'Salário')}", 
                salary_options
            )
            if selected_salaries:
                filters['salary'] = selected_salaries
    
    # Filtro de idade
    if 'age' in df.columns:
        age_min, age_max = int(df['age'].min()), int(df['age'].max())
        if age_min < age_max:
            age_range = st.slider(
                f"🎂 {i18n.t('data.age', 'Idade')}", 
                age_min, age_max, (age_min, age_max)
            )
            if age_range != (age_min, age_max):
                filters['age'] = age_range
    
    # Filtro de sexo
    if 'sex' in df.columns:
        sex_options = [str(x) for x in df['sex'].unique() if str(x) != 'Unknown' and pd.notna(x)]
        if sex_options:
            selected_sex = st.multiselect(
                f"👥 {i18n.t('data.sex', 'Sexo')}", 
                sex_options
            )
            if selected_sex:
                filters['sex'] = selected_sex
    
    # Filtro de educação
    if 'education' in df.columns:
        education_options = [str(x) for x in df['education'].unique() if str(x) != 'Unknown' and pd.notna(x)]
        if education_options:
            selected_education = st.multiselect(
                f"🎓 {i18n.t('data.education', 'Educação')}", 
                education_options
            )
            if selected_education:
                filters['education'] = selected_education
    
    # Mostrar filtros ativos
    if filters:
        st.markdown(f"### 📋 {i18n.t('filters.active', 'Filtros Ativos')}")
        for filter_name, filter_value in filters.items():
            if isinstance(filter_value, list):
                st.write(f"• **{filter_name}**: {', '.join(map(str, filter_value))}")
            elif isinstance(filter_value, tuple):
                st.write(f"• **{filter_name}**: {filter_value[0]} - {filter_value[1]}")
            else:
                st.write(f"• **{filter_name}**: {filter_value}")
    
    return filters

def show_system_status(files_status, i18n):
    """Mostrar status do sistema na sidebar"""
    st.markdown("### 📊 Status do Sistema")
    
    # Verificar se pipeline foi executado
    pipeline_executed = len(files_status.get('models', [])) > 0
    
    if pipeline_executed:
        st.markdown('<div class="success-box">✅ Pipeline Executado!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ Execute: python main.py</div>', unsafe_allow_html=True)
    
    # Métricas de arquivos
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎨 Imagens", len(files_status.get('images', [])))
        st.metric("🤖 Modelos", len(files_status.get('models', [])))
    with col2:
        st.metric("📊 Análises", len(files_status.get('analysis', [])))
        if 'df' in files_status:
            st.metric("📋 Registros", f"{len(files_status['df']):,}")

import pandas as pd  # Import necessário para pd.notna()