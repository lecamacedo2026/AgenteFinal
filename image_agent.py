import base64
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Carrega as variáveis de ambiente
load_dotenv()


def normalizar_endpoint(endpoint: str) -> str:
    """Garante que a URL do endpoint termine corretamente com /openai/v1."""
    url = endpoint.strip()
    if not url.endswith("/openai/v1") and not url.endswith("/openai/v1/"):
        if url.endswith("/"):
            url += "openai/v1"
        else:
            url += "/openai/v1"
    return url


def obter_cliente_openai():
    """Inicializa e retorna o cliente OpenAI configurado para o Microsoft Foundry."""
    endpoint = os.getenv("AZURE_AI_ENDPOINT", "").strip()
    # Aceita AZURE_AI_API_KEY ou FOUNDRY_API_KEY
    api_key = (os.getenv("AZURE_AI_API_KEY") or os.getenv("FOUNDRY_API_KEY") or "").strip()

    if not endpoint or not api_key:
        raise ValueError("AZURE_AI_ENDPOINT e AZURE_AI_API_KEY (ou FOUNDRY_API_KEY) devem estar definidos.")

    endpoint_normalizado = normalizar_endpoint(endpoint)

    return OpenAI(
        base_url=endpoint_normalizado,
        api_key=api_key
    )


def analisar_imagem(image_bytes: bytes, mime_type: str) -> dict | None:
    """
    Codifica a imagem em base64 e realiza a chamada ao modelo no Microsoft Foundry.
    Retorna um dicionário com os dados da análise ou None em caso de erro.
    """
    model_name = os.getenv("AZURE_AI_MODEL", "gpt-4o").strip()
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
        client = obter_cliente_openai()

        response = client.chat.completions.create(
            model=model_name,
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
        print(f"Erro no processamento de image.py: {e}")
        return None
