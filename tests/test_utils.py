import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_env_vars():
    """Testa se as variáveis de ambiente estão configuradas corretamente"""
    logger.info("Testando variáveis de ambiente...")
    
    # Carrega variáveis do .env
    load_dotenv()
    
    # Verifica CLOUD_ID
    cloud_id = os.getenv("ELASTIC_CLOUD_ID")
    if cloud_id:
        logger.info("✅ ELASTIC_CLOUD_ID encontrado")
        # Mostra os primeiros e últimos caracteres para verificação
        logger.info(f"ELASTIC_CLOUD_ID: {cloud_id[:5]}...{cloud_id[-5:]}")
    else:
        logger.error("❌ ELASTIC_CLOUD_ID não encontrado!")
    
    # Verifica ELASTIC_API_KEY
    api_key = os.getenv("ELASTIC_API_KEY")
    if api_key:
        logger.info("✅ ELASTIC_API_KEY encontrado")
        # Mostra os primeiros e últimos caracteres para verificação
        logger.info(f"ELASTIC_API_KEY: {api_key[:5]}...{api_key[-5:]}")
    else:
        logger.error("❌ ELASTIC_API_KEY não encontrado!")
    
    # Verifica o path do arquivo .env
    env_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_path):
        logger.info(f"✅ Arquivo .env encontrado em: {env_path}")
    else:
        logger.error(f"❌ Arquivo .env não encontrado em: {env_path}")

if __name__ == "__main__":
    test_env_vars()