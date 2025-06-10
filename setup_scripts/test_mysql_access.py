"""Testar diferentes métodos de acesso ao MySQL - Versão Completa"""

import subprocess
import mysql.connector
from mysql.connector import Error

def test_mysql_methods():
    """Testar diferentes formas de acessar MySQL"""
    print("🔍 Testando métodos de acesso ao MySQL...")
    
    methods = [
        ("root sem senha", "root", ""),
        ("root com senha vazia", "root", ""),
        ("root com senha 'password'", "root", "password"),
        ("root com senha 'root'", "root", "root"),
        ("salary_user configurado", "salary_user", "senha_forte"),  # Adicionado
        ("mysql sem credenciais", "", ""),
    ]
    
    for desc, user, password in methods:
        try:
            print(f"   Testando: {desc}")
            if user:
                # Testar com database específica se for salary_user
                database = "salary_analysis" if user == "salary_user" else None
                
                conn = mysql.connector.connect(
                    host='localhost',
                    user=user,
                    password=password,
                    database=database,
                    connection_timeout=3
                )
            else:
                # Teste via comando do sistema
                result = subprocess.run(['mysql', '-e', 'SELECT 1;'], 
                                      capture_output=True, timeout=3)
                if result.returncode == 0:
                    print(f"   ✅ {desc} FUNCIONA!")
                    return "system_mysql"
                continue
            
            if conn.is_connected():
                print(f"   ✅ {desc} FUNCIONA!")
                
                # Verificar privilégios
                cursor = conn.cursor()
                cursor.execute("SELECT DATABASE()")
                current_db = cursor.fetchone()
                print(f"      📋 Database atual: {current_db[0] if current_db[0] else 'Nenhuma'}")
                
                cursor.close()
                conn.close()
                return (user, password)
                
        except Error as e:
            print(f"   ❌ {desc}: {e}")
        except subprocess.TimeoutExpired:
            print(f"   ⏱️ {desc}: timeout")
        except Exception as e:
            print(f"   ❌ {desc}: {e}")
    
    # Testar sudo mysql (comum no Ubuntu)
    try:
        print("   Testando: sudo mysql")
        result = subprocess.run(['sudo', 'mysql', '-e', 'SELECT 1;'], 
                              capture_output=True, timeout=5)
        if result.returncode == 0:
            print("   ✅ sudo mysql FUNCIONA!")
            return "sudo_mysql"
    except:
        print("   ❌ sudo mysql: falhou")
    
    return None

def test_project_database():
    """Testar especificamente a base de dados do projeto"""
    print("\n🗄️ Testando base de dados do projeto...")
    
    try:
        conn = mysql.connector.connect(
            host='localhost',
            database='salary_analysis',
            user='salary_user',
            password='senha_forte',
            connection_timeout=3
        )
        
        if conn.is_connected():
            print("   ✅ Conexão com 'salary_analysis' funcionando!")
            
            cursor = conn.cursor()
            
            # Verificar se pode criar tabelas
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS test_connection (
                        id INT PRIMARY KEY,
                        test_message VARCHAR(50)
                    )
                """)
                print("   ✅ Pode criar tabelas")
                
                # Inserir dados de teste
                cursor.execute("INSERT IGNORE INTO test_connection VALUES (1, 'Conexão OK')")
                
                # Ler dados
                cursor.execute("SELECT * FROM test_connection WHERE id = 1")
                result = cursor.fetchone()
                if result:
                    print(f"   ✅ Pode inserir/ler dados: {result[1]}")
                
                # Limpar teste
                cursor.execute("DROP TABLE test_connection")
                print("   ✅ Pode remover tabelas")
                
                conn.commit()
                
            except Error as e:
                print(f"   ⚠️ Limitações de permissão: {e}")
            
            cursor.close()
            conn.close()
            return True
    
    except Error as e:
        print(f"   ❌ Erro na conexão do projeto: {e}")
        return False

def show_next_steps():
    """Mostrar próximos passos"""
    print("\n🚀 PRÓXIMOS PASSOS:")
    print("=" * 30)
    print("1. python setup_sql_only.py    # Verificar configuração completa")
    print("2. python main.py --setup-db    # Criar estrutura de tabelas")
    print("3. python main.py --migrate     # Migrar dados CSV para BD")
    print("4. python main.py               # Executar pipeline completo")
    print("5. streamlit run app.py         # Abrir dashboard")
    
    print("\n📋 Comandos úteis MySQL:")
    print("• Conectar: mysql -u salary_user -p salary_analysis")
    print("• Ver tabelas: SHOW TABLES;")
    print("• Ver dados: SELECT COUNT(*) FROM person;")

def main():
    """Função principal"""
    print("🗄️ TESTE COMPLETO DE ACESSO MYSQL")
    print("=" * 40)
    
    # Testar métodos gerais
    working_method = test_mysql_methods()
    
    if working_method:
        print(f"\n✅ Método funcionando encontrado: {working_method}")
        
        if working_method == "sudo_mysql":
            print("\n💡 Use sudo para acessar MySQL:")
            print("   sudo mysql")
        elif working_method == "system_mysql":
            print("\n💡 Use mysql direto:")
            print("   mysql")
        else:
            user, password = working_method
            print(f"\n💡 Use estas credenciais:")
            print(f"   User: {user}")
            print(f"   Password: {password}")
            print(f"   Comando: mysql -u {user} -p")
            
            # Se for o usuário do projeto, testar database específica
            if user == "salary_user":
                if test_project_database():
                    print("\n🎉 CONFIGURAÇÃO DO PROJETO PERFEITA!")
                    show_next_steps()
                else:
                    print("\n⚠️ Usuário funciona, mas há problemas com a database")
    else:
        print("\n❌ Nenhum método funcionou")
        print("💡 MySQL pode não estar instalado ou rodando")

if __name__ == "__main__":
    main()