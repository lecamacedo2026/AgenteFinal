import streamlit as st
from PIL import Image

from image_agent import ImageAnalysisAgent

st.set_page_config(
    page_title="Agente de Análise de Imagens",
    page_icon="🖼️",
    layout="centered",
)

st.title("🖼️ Agente de Análise de Imagens")
st.caption("Python + Streamlit + Microsoft Foundry")

st.write(
    "Envie uma imagem para que o agente analise objetos, pessoas, cores "
    "e qualidade visual."
)


@st.cache_resource
def load_agent():
    return ImageAnalysisAgent()


try:
    agent = load_agent()
except Exception as exc:
    st.error(f"Erro ao inicializar o agente: {exc}")
    st.info(
        "Confira AZURE_AI_ENDPOINT, AZURE_AI_API_KEY e AZURE_AI_MODEL "
        "no arquivo .env."
    )
    st.stop()


uploaded_file = st.file_uploader(
    "Selecione uma imagem",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded_file is None:
    st.info("Envie uma imagem para iniciar a análise.")
    st.stop()

image_bytes = uploaded_file.getvalue()
mime_type = uploaded_file.type or "image/jpeg"

try:
    image = Image.open(uploaded_file)
    width, height = image.size

    st.image(
        image,
        caption=f"{uploaded_file.name} • {width}x{height}px",
        use_container_width=True,
    )
except Exception as exc:
    st.error(f"Não foi possível abrir a imagem: {exc}")
    st.stop()


if st.button(
    "🔍 Analisar imagem com IA",
    type="primary",
    use_container_width=True,
):
    try:
        with st.spinner("Analisando imagem no Microsoft Foundry..."):
            result = agent.analyze(
                image_bytes=image_bytes,
                mime_type=mime_type,
            )

        st.success("Análise concluída.")

        col1, col2, col3 = st.columns(3)

        possui_pessoas = result.get("possui_pessoas", False)

        with col1:
            st.metric(
                "Pessoas",
                "Sim" if possui_pessoas else "Não",
            )

        with col2:
            st.metric(
                "Quantidade",
                result.get("quantidade_pessoas", 0),
            )

        with col3:
            st.metric(
                "Qualidade",
                str(result.get("qualidade", "indeterminado")).upper(),
            )

        st.subheader("Descrição")
        st.write(result.get("descricao", "Não informado"))

        st.subheader("Elementos identificados")
        elementos = result.get("elementos", [])
        st.write(" • ".join(elementos) if elementos else "Nenhum")

        st.subheader("Cores predominantes")
        cores = result.get("cores_predominantes", [])
        st.write(" • ".join(cores) if cores else "Indeterminado")

        st.subheader("Avaliação técnica")
        st.write(f"**Nitidez:** {result.get('nitidez', 'indeterminado')}")
        st.write(f"**Iluminação:** {result.get('iluminacao', 'indeterminado')}")
        st.write(f"**Contraste:** {result.get('contraste', 'indeterminado')}")
        st.write(f"**Justificativa:** {result.get('reasoning', 'Não informado')}")

        score = float(result.get("score", 0.0))
        st.progress(
            max(0, min(100, int(score * 100))),
            text=f"Confiança da análise: {score * 100:.1f}%",
        )

        with st.expander("Ver JSON retornado"):
            st.json(result)

    except Exception as exc:
        st.error("Erro ao consultar o modelo do Microsoft Foundry.")
        st.exception(exc)
