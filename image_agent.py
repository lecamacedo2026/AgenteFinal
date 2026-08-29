import os
import base64
import json
import re
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# PROMPT DO AGENTE
# ============================================================

SYSTEM_PROMPT = """
[ROLE / PAPEL]

Você é um agente especialista em Processamento de Imagens,
Visão Computacional e Análise Visual de alta precisão,
operando no Microsoft Foundry.

Sua função é analisar exclusivamente as informações
visualmente observáveis na imagem fornecida pelo usuário.


[TASK / TAREFA]

Analise cuidadosamente a imagem e execute as etapas abaixo.


1. DESCRIÇÃO DA IMAGEM

Identifique objetivamente o conteúdo principal da imagem.

Descreva, quando existirem:

- cenário;
- objetos;
- animais;
- veículos;
- equipamentos;
- textos visíveis;
- personagens;
- pessoas reais;
- outros elementos relevantes.

Não invente elementos que não estejam visíveis.


2. DETECÇÃO DE PESSOAS

Determine se existem PESSOAS REAIS visíveis na imagem.

Considere como pessoa real:

- fotografia de uma pessoa;
- pessoa capturada por câmera;
- rosto humano real;
- corpo humano real visível.

NÃO considere como pessoa real:

- personagens de anime;
- desenhos;
- ilustrações;
- pinturas;
- estátuas;
- bonecos;
- avatares;
- personagens de videogame;
- personagens 3D;
- personagens fictícios;
- imagens claramente artificiais.

Se existirem somente personagens desenhados,
digitais ou fictícios, retorne:

"possui_pessoas": false

e:

"quantidade_pessoas": 0

Os personagens devem ser descritos normalmente
nos campos:

- descricao
- elementos
- keywords

Caso existam pessoas reais, informe apenas
a quantidade aproximada.

Nunca tente identificar quem são as pessoas.


3. ELEMENTOS

Liste os principais elementos detectados.

Exemplo:

[
    "pessoa",
    "computador",
    "mesa",
    "cadeira"
]

Outro exemplo:

[
    "personagens de anime",
    "armaduras douradas",
    "elmos",
    "fundo escuro"
]

Utilize descrições curtas e objetivas.


4. CORES PREDOMINANTES

Identifique somente as principais cores
visualmente predominantes na imagem.

Exemplo:

[
    "preto",
    "dourado",
    "azul"
]

Não liste cores pouco relevantes.


5. QUALIDADE DA IMAGEM

Avalie tecnicamente:

- nitidez;
- iluminação;
- contraste;
- presença de ruído;
- presença de desfoque;
- definição visual;
- resolução aparente;
- capacidade de distinguir detalhes.

Classifique a qualidade geral utilizando
SOMENTE uma destas opções:

"alta"
"media"
"baixa"
"indeterminado"

Utilize "media" sem acento.


6. NITIDEZ

Classifique utilizando SOMENTE:

"alta"
"media"
"baixa"
"indeterminado"


7. ILUMINAÇÃO

Classifique utilizando SOMENTE:

"boa"
"regular"
"ruim"
"indeterminado"


8. CONTRASTE

Classifique utilizando SOMENTE:

"bom"
"regular"
"ruim"
"indeterminado"


9. SCORE

O campo "score" represents a confiança geral
da análise realizada pelo modelo.

Utilize um número decimal entre 0 e 1.

Exemplos:

0.98 = confiança muito alta
0.80 = confiança alta
0.60 = confiança moderada
0.30 = confiança baixa

IMPORTANTE:

O score representa CONFIANÇA DA ANÁLISE.

O score NÃO representa a qualidade da imagem.


10. KEYWORDS

Forneça palavras-chave relacionadas aos
principais elementos visuais encontrados.

Exemplo:

[
    "anime",
    "armaduras",
    "dourado",
    "personagens"
]


11. EXPLICAÇÃO TÉCNICA

O campo "reasoning" deve apresentar uma
justificativa curta sobre a qualidade visual.

A explicação deve conter no máximo 15 palavras.


[REGRAS IMPORTANTES]

- Analise somente informações visualmente observáveis.
- Não invente informações.
- Não identifique pessoas.
- Não tente descobrir nomes de pessoas.
- Não faça suposições sobre identidade.
- Não faça suposições sobre profissão.
- Não faça suposições sobre personalidade.
- Diferencie pessoas reais de personagens fictícios.
- Personagens de anime NÃO são pessoas reais.
- Personagens de videogames NÃO são pessoas reais.
- Ilustrações humanas NÃO são pessoas reais.
- Bonecos NÃO são pessoas reais.
- Estátuas NÃO são pessoas reais.
- Avatares digitais NÃO são pessoas reais.
- Responda em português do Brasil.
- Seja objetivo.
- Utilize exatamente os campos solicitados.
- Não adicione campos diferentes.
""".strip()


# ============================================================
# JSON SCHEMA
# Obriga o modelo a seguir esta estrutura
# ============================================================

IMAGE_ANALYSIS_SCHEMA = {
    "type": "object",

    "properties": {

        "descricao": {
            "type": "string"
        },

        "possui_pessoas": {
            "type": "boolean"
        },

        "quantidade_pessoas": {
            "type": "integer"
        },

        "elementos": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "cores_predominantes": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "qualidade": {
            "type": "string",
            "enum": [
                "alta",
                "media",
                "baixa",
                "indeterminado"
            ]
        },

        "nitidez": {
            "type": "string",
            "enum": [
                "alta",
                "media",
                "baixa",
                "indeterminado"
            ]
        },

        "iluminacao": {
            "type": "string",
            "enum": [
                "boa",
                "regular",
                "ruim",
                "indeterminado"
            ]
        },

        "contraste": {
            "type": "string",
            "enum": [
                "bom",
                "regular",
                "ruim",
                "indeterminado"
            ]
        },

        "score": {
            "type": "number"
        },

        "language": {
            "type": "string"
        },

        "keywords": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "reasoning": {
            "type": "string"
        }
    },

    "required": [
        "descricao",
        "possui_pessoas",
        "quantidade_pessoas",
        "elementos",
        "cores_predominantes",
        "qualidade",
        "nitidez",
        "iluminacao",
        "contraste",
        "score",
        "language",
        "keywords",
        "reasoning"
    ],

    "additionalProperties": False
}


# ============================================================
# AGENTE
# ============================================================

class ImageAnalysisAgent:

    def __init__(self):

        # Carrega as variáveis do arquivo .env
        load_dotenv()

        self.endpoint = os.getenv(
            "AZURE_AI_ENDPOINT",
            ""
        ).strip()

        self.api_key = os.getenv(
            "AZURE_AI_API_KEY",
            ""
        ).strip()

        self.model = os.getenv(
            "AZURE_AI_MODEL",
            ""
        ).strip()


        # ----------------------------------------------------
        # VALIDAÇÃO DAS VARIÁVEIS
        # ----------------------------------------------------

        if not self.endpoint:
            raise ValueError(
                "AZURE_AI_ENDPOINT não foi definido "
                "no arquivo .env."
            )

        if not self.api_key:
            raise ValueError(
                "AZURE_AI_API_KEY não foi definida "
                "no arquivo .env."
            )

        if not self.model:
            raise ValueError(
                "AZURE_AI_MODEL não foi definido. "
                "Informe o nome do deployment multimodal "
                "publicado no Microsoft Foundry."
            )


        # ----------------------------------------------------
        # NORMALIZA ENDPOINT
        # ----------------------------------------------------

        self.endpoint = self._normalize_endpoint(
            self.endpoint
        )


        # ----------------------------------------------------
        # CRIA CLIENTE
        # ----------------------------------------------------

        self.client = OpenAI(
            base_url=self.endpoint,
            api_key=self.api_key
        )

    def _normalize_endpoint(self, url: str) -> str:
        """Garante que a URL termine corretamente com a rota do gateway."""
        url_limpa = url.strip()
        if not url_limpa.endswith("/openai/v1") and not url_limpa.endswith("/openai/v1/"):
            if url_limpa.endswith("/"):
                url_limpa = url_limpa + "openai/v1"
            else:
                url_limpa = url_limpa + "/openai/v1"
        return url_limpa

    def _limpar_resposta_json(self, texto_cru: str) -> str:
        """Filtra wrappers Markdown da resposta."""
        texto_limpo = texto_cru.strip()
        texto_limpo = re.sub(r'^```json\s*', '', texto_limpo, flags=re.IGNORECASE)
        texto_limpo = re.sub(r'^```\s*', '', texto_limpo)
        texto_limpo = re.sub(r'\s*```$', '', texto_limpo)
        return texto_limpo.strip()

    def analyze(self, image_bytes: bytes, mime_type: str) -> dict:
        """Converte a imagem, faz a chamada estruturada e valida a resposta."""
        
        # Converte os bytes em string base64 legível por IA
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_data_uri = f"data:{mime_type};base64,{base64_image}"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analise esta imagem estritamente conforme as regras do sistema."},
                    {"type": "image_url", "image_url": {"url": image_data_uri}}
                ]
            }
        ]
        
        # Injeta o SCHEMA usando o padrão estrito de JSON Object da OpenAI compatível com o Foundry
