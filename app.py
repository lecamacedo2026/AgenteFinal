import base64
import json
import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env (caso exista localmente)
load_dotenv()

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Agente de Análise de Imagens",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Agente de Análise de Imagens")
st.caption("Python + Streamlit + Microsoft Foundry")
st.write("Envie uma imagem para que o agente analise objetos, pessoas, núcleos e qualidade visual.")

# ------------------------------------------------------------------------------
# 1. Carregamento e validação das variáveis de ambiente
# ------------------------------------------------------------------------------
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT", "").strip()
# Aceita AZURE_AI_API_KEY ou FOUNDRY_API_KEY como fallback
AZURE_AI_API_KEY = (os.getenv("AZURE_AI_API_KEY") or os.getenv("FOUNDRY_API_KEY") or "").strip()
AZURE_AI_MODEL = os.getenv("AZURE_AI_MODEL", "gpt-4o").strip()

if not AZURE_AI_API_KEY or not AZURE_AI_ENDPOINT:
    st.error("Erro ao inicializar o agente: AZURE_AI_API_KEY (ou FOUNDRY_API_KEY) ou AZURE_AI_ENDPOINT não foi definido no ambiente / arquivo .env.")
    st.info("Confira AZURE_AI_ENDPOINT, AZURE_AI_API_KEY e AZURE_AI_MODEL no arquivo .env ou nas variáveis do servidor.")
    st.stop()

# Garantir que o endpoint termine com /openai/v1 para uso com o SDK OpenAI
if not AZURE_AI_ENDPOINT.endswith("/openai/v1") and not AZURE_AI_ENDPOINT.endswith("/openai/v1/"):
    if AZURE_AI_ENDPOINT.endswith("/"):
        AZURE_AI_ENDPOINT += "openai/v1"
    else:
        AZURE_AI_ENDPOINT += "/openai/v1"

# ------------------------------------------------------------------------------
# 2. Inicialização do Cliente OpenAI (Compatível com Microsoft Foundry /v1)
# ------------------------------------------------------------------------------
try:
    client = OpenAI(
        base_url=AZURE_AI_ENDPOINT,
        api_key=AZURE_AI_API_KEY
    )
except Exception as e:
    st.error(f"Erro ao criar o cliente do Azure/Foundry AI: {e}")
    st.stop()


# ------------------------------------------------------------------------------
# 3. Função de Análise de Imagem (com tratamento rigoroso contra None)
# ------------------------------------------------------------------------------
def analisar_imagem(image_bytes, mime_type):
    """Envia a imagem codificada em base64 para o modelo no Microsoft Foundry e retorna um dicionário JSON."""
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    prompt = """
    Analise a imagem fornecida e retorne estritamente um JSON no seguinte formato:
    {
        "possui_pessoas": true/false,
        "objetos": ["objeto1", "objeto2"],
        "nucleos_predominantes": ["cor1", "cor2"],
        "qualidade_visual": "Boa / Regular / Ruim",
        "descricao": "Breve resumo descritivo da imagem"
    }
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_AI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=800
        )
        
        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        st.error(f"Erro ao consultar o modelo do Microsoft Foundry: {e}")
        return None


# ------------------------------------------------------------------------------
# 4. Interface da aplicação
# ------------------------------------------------------------------------------
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    st.image(uploaded_file, caption=f"{uploaded_file.name}", use_container_width=True)
    
    if st.button("🔍 Analisar imagem com IA", type="primary"):
        with st.spinner("Analisando imagem..."):
            image_bytes = uploaded_file.getvalue()
            mime_type = uploaded_file.type or "image/jpeg"
            
            # Chamada da API
            result = analisar_imagem(image_bytes, mime_type)
            
            # Validação para evitar AttributeError: 'NoneType' object has no attribute 'get'
            if result is not None and isinstance(result, dict):
                st.success("Análise concluída.")
                
                # Leitura segura das chaves com .get()
                possui_pessoas = result.get("possui_pessoas", False)
                objetos = result.get("objetos", [])
                nucleos = result.get("nucleos_predominantes", [])
                qualidade = result.get("qualidade_visual", "Não informada")
                descricao = result.get("descricao", "Sem descrição.")
                
                # Exibição dos resultados
                st.subheader("Resultados da Análise")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Possui pessoas:** {'Sim' if possui_pessoas else 'Não'}")
                    st.write(f"**Qualidade visual:** {qualidade}")
                
                with col2:
                    st.write(f"**Cores predominantes:** {', '.join(nucleos) if nucleos else 'N/A'}")
                    st.write(f"**Objetos identificados:** {', '.join(objetos) if objetos else 'N/A'}")
                
                st.info(f"**Descrição:** {descricao}")
            else:
                st.error("Não foi possível processar a imagem devido a uma falha na resposta da API.")
