"""
🔐 Página de Login
Interface de autenticação multilingual
"""

import streamlit as st
import time

def show_login_page(auth_system, i18n):
    """Página de login multilingual"""
    
    # Header da aplicação
    st.markdown(f'<div class="login-title">{i18n.t("app.title", "💰 Dashboard Salarial")}</div>', 
                unsafe_allow_html=True)
    
    # Mostrar seletor de idioma no topo
    i18n.show_language_selector()
    
    # Container principal centralizado
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            
            # Tabs de login e registro
            tab1, tab2 = st.tabs([
                f"🔐 {i18n.t('auth.login_button', 'Login')}", 
                f"📝 {i18n.t('auth.register', 'Registro')}"
            ])
            
            with tab1:
                _show_login_form(auth_system, i18n)
            
            with tab2:
                _show_register_form(auth_system, i18n)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Informações de demonstração
            _show_demo_info(i18n)

def _show_login_form(auth_system, i18n):
    """Formulário de login"""
    st.markdown(f'<div class="login-form">', unsafe_allow_html=True)
    
    with st.form("login_form"):
        st.markdown(f"### {i18n.t('auth.login_title', '🔓 Acesso ao Sistema')}")
        
        username = st.text_input(
            i18n.t('auth.username', '👤 Usuário'), 
            placeholder="admin, demo, guest"
        )
        password = st.text_input(
            i18n.t('auth.password', '🔑 Senha'), 
            type="password", 
            placeholder="admin123, demo123, guest123"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            login_button = st.form_submit_button(
                i18n.t('auth.login_button', '🚀 Entrar'), 
                use_container_width=True
            )
        with col2:
            demo_button = st.form_submit_button(
                i18n.t('auth.demo_button', '🎮 Demo'), 
                use_container_width=True
            )
        
        # Processar login
        if demo_button:
            username = "demo"
            password = "demo123"
            login_button = True
        
        if login_button and username and password:
            _process_login(auth_system, i18n, username, password)
    
    st.markdown('</div>', unsafe_allow_html=True)

def _show_register_form(auth_system, i18n):
    """Formulário de registro"""
    st.markdown('<div class="login-form">', unsafe_allow_html=True)
    
    with st.form("register_form"):
        st.markdown(f"### {i18n.t('auth.register', '📝 Criar Conta')}")
        
        new_username = st.text_input(
            i18n.t('auth.username', '👤 Novo Usuário'), 
            placeholder="Escolha um usuário"
        )
        new_name = st.text_input(
            "👨‍💼 Nome Completo", 
            placeholder="Seu nome completo"
        )
        new_email = st.text_input(
            "📧 Email", 
            placeholder="seu@email.com"
        )
        new_password = st.text_input(
            i18n.t('auth.password', '🔑 Senha'), 
            type="password", 
            placeholder="Escolha uma senha"
        )
        confirm_password = st.text_input(
            "🔑 Confirmar Senha", 
            type="password", 
            placeholder="Confirme a senha"
        )
        
        register_button = st.form_submit_button(
            "✨ Criar Conta", 
            use_container_width=True
        )
        
        if register_button:
            _process_registration(auth_system, i18n, new_username, new_name, 
                                new_email, new_password, confirm_password)
    
    st.markdown('</div>', unsafe_allow_html=True)

def _process_login(auth_system, i18n, username, password):
    """Processar tentativa de login"""
    user_data = auth_system.authenticate(username, password)
    
    if user_data:
        # Criar sessão
        session_id = auth_system.create_session(username, user_data)
        
        # Atualizar estado da sessão
        st.session_state.session_id = session_id
        st.session_state.authenticated = True
        st.session_state.user_data = user_data
        st.session_state.username = username
        
        # Mensagem de sucesso
        welcome_msg = i18n.t('auth.welcome', 'Bem-vindo')
        user_name = user_data.get('name', username)
        st.success(f"✅ {welcome_msg} {user_name}!")
        
        time.sleep(1)
        st.rerun()
    else:
        st.error(i18n.t('auth.invalid_credentials', '❌ Credenciais inválidas!'))

def _process_registration(auth_system, i18n, username, name, email, password, confirm_password):
    """Processar registro de novo usuário"""
    # Validações
    if not all([username, name, email, password]):
        st.error("❌ Preencha todos os campos!")
        return
    
    if password != confirm_password:
        st.error("❌ Senhas não coincidem!")
        return
    
    if len(password) < 6:
        st.error("❌ Senha deve ter pelo menos 6 caracteres!")
        return
    
    # Tentar registrar
    try:
        success, message = auth_system.register_user(username, password, name, email)
        
        if success:
            st.success(f"✅ {message}")
            st.info("🔄 Agora você pode fazer login!")
        else:
            st.error(f"❌ {message}")
    except Exception as e:
        st.error(f"❌ Erro no registro: {e}")

def _show_demo_info(i18n):
    """Mostrar informações das contas demo"""
    st.markdown(f"""
    <div class="demo-info">
        <h4>🎮 {i18n.t('auth.demo_button', 'Contas de Demonstração')}:</h4>
        <ul>
            <li><strong>admin</strong> / admin123 - ({i18n.t('auth.role', 'Papel')}: Administrador)</li>
            <li><strong>demo</strong> / demo123 - ({i18n.t('auth.role', 'Papel')}: Usuário)</li>
            <li><strong>guest</strong> / guest123 - ({i18n.t('auth.role', 'Papel')}: Visitante)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)