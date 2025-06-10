"""
🔐 Componente de Login Multilingual
Interface moderna com suporte a múltiplos idiomas
"""

import streamlit as st
import time


def show_login_page(auth_system, i18n):
    """Página de login modernizada com i18n"""
    
    # CSS customizado para login
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        color: white;
    }
    .login-title {
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .login-form {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    .demo-info {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        backdrop-filter: blur(5px);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header da aplicação
    st.markdown(f'<div class="login-title">{i18n.t("app.title")}</div>', unsafe_allow_html=True)
    
    # Container principal
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            
            # Tabs de login e registro
            tab1, tab2 = st.tabs([
                f"🔐 {i18n.t('auth.login_button')}", 
                f"📝 {i18n.t('auth.register')}"
            ])
            
            with tab1:
                st.markdown('<div class="login-form">', unsafe_allow_html=True)
                
                with st.form("login_form"):
                    st.markdown(f"### {i18n.t('auth.login_title')}")
                    
                    username = st.text_input(
                        i18n.t("auth.username"), 
                        placeholder=i18n.t("auth.username")
                    )
                    password = st.text_input(
                        i18n.t("auth.password"), 
                        type="password", 
                        placeholder=i18n.t("auth.password")
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        login_button = st.form_submit_button(
                            i18n.t("auth.login_button"), 
                            use_container_width=True
                        )
                    with col2:
                        demo_button = st.form_submit_button(
                            i18n.t("auth.demo_button"), 
                            use_container_width=True
                        )
                    
                    if login_button and username and password:
                        user_data = auth_system.authenticate(username, password)
                        if user_data:
                            session_id = auth_system.create_session(username, user_data)
                            st.session_state.session_id = session_id
                            st.session_state.user_data = user_data
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            
                            user_name = auth_system.get_user_display_name(user_data)
                            st.success(f"✅ {i18n.t('auth.welcome')}, {user_name}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(i18n.t("auth.invalid_credentials"))
                    
                    if demo_button:
                        # Login automático como demo
                        user_data = auth_system.authenticate("demo", "demo123")
                        if user_data:
                            session_id = auth_system.create_session("demo", user_data)
                            st.session_state.session_id = session_id
                            st.session_state.user_data = user_data
                            st.session_state.authenticated = True
                            st.session_state.username = "demo"
                            st.success(f"✅ {i18n.t('auth.demo_button')}...")
                            time.sleep(1)
                            st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown('<div class="login-form">', unsafe_allow_html=True)
                
                with st.form("register_form"):
                    st.markdown(f"### {i18n.t('auth.register')}")
                    
                    new_username = st.text_input(
                        i18n.t("auth.username"), 
                        placeholder=i18n.t("auth.username")
                    )
                    new_name = st.text_input(
                        "Nome Completo | Full Name", 
                        placeholder="Nome completo | Full name"
                    )
                    new_email = st.text_input(
                        "📧 Email", 
                        placeholder="seu@email.com | your@email.com"
                    )
                    new_password = st.text_input(
                        i18n.t("auth.password"), 
                        type="password", 
                        placeholder=i18n.t("auth.password")
                    )
                    confirm_password = st.text_input(
                        "🔑 Confirmar | Confirm", 
                        type="password", 
                        placeholder="Confirme a senha | Confirm password"
                    )
                    
                    register_button = st.form_submit_button(
                        f"✨ {i18n.t('auth.register')}", 
                        use_container_width=True
                    )
                    
                    if register_button:
                        if not all([new_username, new_name, new_email, new_password]):
                            st.error("❌ Preencha todos os campos | Fill all fields!")
                        elif new_password != confirm_password:
                            st.error("❌ Senhas não coincidem | Passwords don't match!")
                        elif len(new_password) < 6:
                            st.error("❌ Senha deve ter 6+ caracteres | Password must be 6+ chars!")
                        else:
                            success, message = auth_system.register_user(
                                new_username, new_password, new_name, new_email
                            )
                            if success:
                                st.success(message)
                                st.info("🔄 Agora você pode fazer login | Now you can login!")
                            else:
                                st.error(message)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Informações de demo
            st.markdown(f"""
            <div class="demo-info">
                <h4>🎮 {i18n.t("auth.demo_button")}:</h4>
                <ul>
                    <li><strong>admin</strong> / admin123 ({i18n.t("auth.role")}: admin)</li>
                    <li><strong>demo</strong> / demo123 ({i18n.t("auth.role")}: user)</li>
                    <li><strong>guest</strong> / guest123 ({i18n.t("auth.role")}: guest)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)


def show_user_info(auth_system, i18n):
    """Mostrar informações do usuário na sidebar"""
    if 'user_data' in st.session_state and st.session_state.user_data:
        user = st.session_state.user_data
        username = st.session_state.get('username', 'N/A')
        
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"### {i18n.t('auth.logged_user')}")
            
            user_name = auth_system.get_user_display_name(user)
            
            st.markdown(f"""
            <div class="user-info-card">
                <h4>🎭 {user_name}</h4>
                <p><strong>{i18n.t("auth.username")}:</strong> {username}</p>
                <p><strong>{i18n.t("auth.role")}:</strong> {user['role'].title()}</p>
                <p><strong>📧 Email:</strong> {user['email']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(i18n.t("auth.logout"), use_container_width=True):
                if st.session_state.session_id:
                    auth_system.logout(st.session_state.session_id)
                
                # Limpar sessão
                st.session_state.authenticated = False
                st.session_state.session_id = None
                st.session_state.user_data = None
                st.session_state.username = None
                st.session_state.current_page = i18n.t("navigation.overview")
                
                st.success(i18n.t("auth.logout_success"))
                time.sleep(1)
                st.rerun()