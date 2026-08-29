import base64
import json
import os
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

O campo "score" representa a confiança geral
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
            api_key=self.api_key,
            base_url=self.endpoint,
        )


    # ========================================================
    # NORMALIZA ENDPOINT
    # ========================================================

    @staticmethod
    def _normalize_endpoint(
        endpoint: str
    ) -> str:

        endpoint = endpoint.strip().rstrip("/")

        # Caso já esteja no formato correto
        if endpoint.endswith("/openai/v1"):

            return endpoint + "/"

        # Acrescenta a rota da API OpenAI v1
        return endpoint + "/openai/v1/"


    # ========================================================
    # CONVERTE IMAGEM PARA BASE64
    # ========================================================

    @staticmethod
    def _to_data_url(
        image_bytes: bytes,
        mime_type: str,
    ) -> str:

        encoded = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        return (
            f"data:{mime_type};base64,{encoded}"
        )


    # ========================================================
    # VALIDA RESULTADO
    # ========================================================

    @staticmethod
    def _validate(
        data: dict[str, Any]
    ) -> dict[str, Any]:


        # ----------------------------------------------------
        # QUALIDADE
        # ----------------------------------------------------

        qualidade = str(
            data.get(
                "qualidade",
                "indeterminado"
            )
        ).strip().lower()

        if qualidade == "média":
            qualidade = "media"

        if qualidade not in {
            "alta",
            "media",
            "baixa",
            "indeterminado",
        }:

            qualidade = "indeterminado"


        # ----------------------------------------------------
        # PESSOAS
        # ----------------------------------------------------

        possui_pessoas = bool(
            data.get(
                "possui_pessoas",
                False
            )
        )


        try:

            quantidade = int(
                data.get(
                    "quantidade_pessoas",
                    0
                )
            )

        except (TypeError, ValueError):

            quantidade = 0


        if not possui_pessoas:

            quantidade = 0


        quantidade = max(
            0,
            quantidade
        )


        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        try:

            score = float(
                data.get(
                    "score",
                    0.5
                )
            )

        except (TypeError, ValueError):

            score = 0.5


        score = max(
            0.0,
            min(
                1.0,
                score
            )
        )


        # ----------------------------------------------------
        # ELEMENTOS
        # ----------------------------------------------------

        elementos = data.get(
            "elementos",
            []
        )

        if not isinstance(
            elementos,
            list
        ):

            elementos = [
                str(elementos)
            ]


        # ----------------------------------------------------
        # CORES
        # ----------------------------------------------------

        cores = data.get(
            "cores_predominantes",
            []
        )

        if not isinstance(
            cores,
            list
        ):

            cores = [
                str(cores)
            ]


        # ----------------------------------------------------
        # KEYWORDS
        # ----------------------------------------------------

        keywords = data.get(
            "keywords",
            []
        )

        if not isinstance(
            keywords,
            list
        ):

            keywords = [
                str(keywords)
            ]


        # ----------------------------------------------------
        # REASONING
        # Máximo 15 palavras
        # ----------------------------------------------------

        reasoning = str(
            data.get(
                "reasoning",
                "Análise técnica não informada."
            )
        ).strip()


        words = reasoning.split()


        if len(words) > 15:

            reasoning = " ".join(
                words[:15]
            )


        # ----------------------------------------------------
        # RESULTADO FINAL
        # ----------------------------------------------------

        return {

            "descricao": str(
                data.get(
                    "descricao",
                    "Descrição não informada."
                )
            ).strip(),

            "possui_pessoas": possui_pessoas,

            "quantidade_pessoas": quantidade,

            "elementos": elementos,

            "cores_predominantes": cores,

            "qualidade": qualidade,

            "nitidez": str(
                data.get(
                    "nitidez",
                    "indeterminado"
                )
            ).strip(),

            "iluminacao": str(
                data.get(
                    "iluminacao",
                    "indeterminado"
                )
            ).strip(),

            "contraste": str(
                data.get(
                    "contraste",
                    "indeterminado"
                )
            ).strip(),

            "score": score,

            "language": "pt",

            "keywords": keywords,

            "reasoning": reasoning,
        }


    # ========================================================
    # ANALISA A IMAGEM
    # ========================================================

    def analyze(
        self,
        image_bytes: bytes,
        mime_type: str,
    ) -> dict[str, Any]:


        if not image_bytes:

            raise ValueError(
                "A imagem está vazia."
            )


        # ----------------------------------------------------
        # CONVERTE PARA BASE64
        # ----------------------------------------------------

        data_url = self._to_data_url(

            image_bytes=image_bytes,

            mime_type=mime_type,
        )


        # ----------------------------------------------------
        # CHAMADA AO MICROSOFT FOUNDRY
        # ----------------------------------------------------

        response = self.client.responses.create(

            model=self.model,

            instructions=SYSTEM_PROMPT,

            input=[
                {
                    "type": "message",

                    "role": "user",

                    "content": [

                        {
                            "type": "input_text",

                            "text": (
                                "Analise cuidadosamente a imagem "
                                "fornecida conforme todas as instruções."
                            ),
                        },

                        {
                            "type": "input_image",

                            "image_url": data_url,

                            "detail": "high",
                        },
                    ],
                }
            ],


            # ------------------------------------------------
            # STRUCTURED OUTPUT
            # Obriga o modelo a retornar o JSON no schema
            # ------------------------------------------------

            text={
                "format": {

                    "type": "json_schema",

                    "name": "analise_imagem",

                    "strict": True,

                    "schema": IMAGE_ANALYSIS_SCHEMA,
                }
            },


            # ------------------------------------------------
            # LIMITE DE SAÍDA
            # Antes estava 1000 e o JSON foi cortado
            # ------------------------------------------------

            max_output_tokens=4000,
        )


        # ----------------------------------------------------
        # VERIFICA SE A RESPOSTA FOI INTERROMPIDA
        # ----------------------------------------------------

        status = getattr(
            response,
            "status",
            None
        )


        if status == "incomplete":

            details = getattr(
                response,
                "incomplete_details",
                None
            )

            raise ValueError(
                "O Microsoft Foundry interrompeu "
                "a resposta antes de finalizar. "
                f"Detalhes: {details}"
            )


        # ----------------------------------------------------
        # TEXTO RETORNADO
        # ----------------------------------------------------

        content = response.output_text


        if not content:

            raise ValueError(
                "O modelo retornou uma resposta vazia. "
                "Confirme se o deployment suporta imagens."
            )


        # ----------------------------------------------------
        # CONVERTE O JSON
        # ----------------------------------------------------

        try:

            parsed = json.loads(
                content
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                "O modelo retornou um JSON inválido "
                "ou incompleto. "
                f"Resposta recebida: {content}"
            ) from exc


        # ----------------------------------------------------
        # VALIDA E RETORNA
        # ----------------------------------------------------

        return self._validate(
            parsed
        )