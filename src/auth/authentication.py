"""
🔐 Sistema de Autenticação Multilingual
Gerenciamento completo de usuários e sessões
"""

import streamlit as st
import hashlib
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AuthenticationSystem:
    """Sistema de autenticação moderno com suporte multilingual"""
    
    def __init__(self, i18n_system):
        self.i18n = i18n_system
        self.users_file = Path("config/users.json")
        self.sessions_file = Path("config/sessions.json")
        self._init_files()
        
        # Usuários padrão
        self.default_users = {
            "admin": {
                "password": self.hash_password("admin123"),
                "role": "admin",
                "name": "Administrador",
                "email": "admin@dashboard.com",
                "created": datetime.now().isoformat()
            },
            "demo": {
                "password": self.hash_password("demo123"),
                "role": "user", 
                "name": "Usuário Demo",
                "email": "demo@dashboard.com",
                "created": datetime.now().isoformat()
            },
            "guest": {
                "password": self.hash_password("guest123"),
                "role": "guest",
                "name": "Visitante",
                "email": "guest@dashboard.com", 
                "created": datetime.now().isoformat()
            }
        }
        
        self._ensure_default_users()
    
    def _init_files(self):
        """Inicializar arquivos de configuração"""
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        for file_path in [self.users_file, self.sessions_file]:
            if not file_path.exists():
                file_path.write_text("{}", encoding='utf-8')
                logger.info(f"✅ Arquivo criado: {file_path}")
    
    def hash_password(self, password):
        """Hash seguro da senha"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def load_users(self):
        """Carregar usuários do arquivo"""
        try:
            return json.loads(self.users_file.read_text(encoding='utf-8'))
        except Exception as e:
            logger.error(f"Erro ao carregar usuários: {e}")
            return {}
    
    def save_users(self, users):
        """Salvar usuários no arquivo"""
        try:
            self.users_file.write_text(json.dumps(users, indent=2), encoding='utf-8')
            logger.info("✅ Usuários salvos")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar usuários: {e}")
            st.error(f"Erro ao salvar usuários: {e}")
    
    def _ensure_default_users(self):
        """Garantir que usuários padrão existam"""
        users = self.load_users()
        updated = False
        
        for username, user_data in self.default_users.items():
            if username not in users:
                users[username] = user_data
                updated = True
                logger.info(f"✅ Usuário padrão criado: {username}")
        
        if updated:
            self.save_users(users)
    
    def authenticate(self, username, password):
        """Autenticar usuário"""
        users = self.load_users()
        
        if username in users:
            stored_password = users[username]["password"]
            if stored_password == self.hash_password(password):
                logger.info(f"✅ Login bem-sucedido: {username}")
                return users[username]
        
        logger.warning(f"❌ Falha na autenticação: {username}")
        return None
    
    def create_session(self, username, user_data):
        """Criar sessão de usuário"""
        session_id = hashlib.md5(f"{username}{time.time()}".encode()).hexdigest()
        
        try:
            sessions = json.loads(self.sessions_file.read_text(encoding='utf-8'))
        except:
            sessions = {}
        
        sessions[session_id] = {
            "username": username,
            "user_data": user_data,
            "created": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
        
        # Limpar sessões antigas (>24h)
        cutoff = datetime.now() - timedelta(hours=24)
        sessions = {
            sid: sdata for sid, sdata in sessions.items()
            if datetime.fromisoformat(sdata["last_activity"]) > cutoff
        }
        
        try:
            self.sessions_file.write_text(json.dumps(sessions, indent=2), encoding='utf-8')
        except Exception as e:
            logger.error(f"Erro ao salvar sessão: {e}")
        
        logger.info(f"✅ Sessão criada: {username}")
        return session_id
    
    def get_session(self, session_id):
        """Obter dados da sessão"""
        try:
            sessions = json.loads(self.sessions_file.read_text(encoding='utf-8'))
            if session_id in sessions:
                session = sessions[session_id]
                # Verificar se não expirou
                last_activity = datetime.fromisoformat(session["last_activity"])
                if datetime.now() - last_activity < timedelta(hours=24):
                    # Atualizar última atividade
                    session["last_activity"] = datetime.now().isoformat()
                    sessions[session_id] = session
                    self.sessions_file.write_text(json.dumps(sessions, indent=2), encoding='utf-8')
                    return session
        except Exception as e:
            logger.error(f"Erro ao obter sessão: {e}")
        
        return None
    
    def logout(self, session_id):
        """Fazer logout"""
        try:
            sessions = json.loads(self.sessions_file.read_text(encoding='utf-8'))
            if session_id in sessions:
                username = sessions[session_id].get('username', 'Unknown')
                del sessions[session_id]
                self.sessions_file.write_text(json.dumps(sessions, indent=2), encoding='utf-8')
                logger.info(f"✅ Logout: {username}")
        except Exception as e:
            logger.error(f"Erro no logout: {e}")
    
    def register_user(self, username, password, name, email):
        """Registrar novo usuário"""
        users = self.load_users()
        
        # Verificar se usuário já existe
        if username in users:
            return False, "Usuário já existe!"
        
        # Verificar se email já existe
        for existing_user in users.values():
            if existing_user.get('email') == email:
                return False, "Email já está em uso!"
        
        # Criar novo usuário
        new_user = {
            "password": self.hash_password(password),
            "role": "user",  # Papel padrão
            "name": name,
            "email": email,
            "created": datetime.now().isoformat()
        }
        
        # Adicionar aos usuários
        users[username] = new_user
        
        # Salvar
        try:
            self.save_users(users)
            logger.info(f"✅ Novo usuário registrado: {username}")
            return True, "Usuário criado com sucesso!"
        except Exception as e:
            logger.error(f"❌ Erro ao registrar usuário: {e}")
            return False, f"Erro ao salvar usuário: {e}"
    
    def get_user_display_name(self, user_data):
        """Obter nome de exibição do usuário baseado no idioma"""
        name = user_data.get('name', 'Unknown')
        
        if isinstance(name, dict):
            current_lang = self.i18n.get_language()
            return name.get(current_lang, name.get('en', str(name)))
        
        return str(name)