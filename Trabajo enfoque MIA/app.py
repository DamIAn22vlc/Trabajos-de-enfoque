import streamlit as st
from EnfoqueMIA import client, cargar_base_conocimiento, cargar_modelo_entrenado

st.set_page_config(page_title="Asistente Soporte Técnico", page_icon="🤖")
st.title("🤖 Soporte Técnico IA")
st.markdown("---")


with st.sidebar:
    st.subheader("⚖️ Transparencia y Privacidad")
    st.info(
        """
    **Aviso Legal:**
    1. Este asistente es una **IA local**. Sus datos no salen de este equipo.
    2. Las respuestas se basan en un manual técnico predefinido.
    3. Si el sistema no está seguro, le derivará a un técnico humano.
    """
    )
    if st.button("Borrar historial de datos"):
        st.session_state.messages = []
        st.success("Datos de sesión eliminados.")


@st.cache_resource
def load_assets():
    modelo = cargar_modelo_entrenado()
    datos = cargar_base_conocimiento()
    return modelo, datos


modelo_entrenado, base_datos = load_assets()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        categoria = modelo_entrenado.predict([prompt])[0]
        st.caption(f"Categoría detectada: {categoria}")

        contexto = str(base_datos)
        response = client.chat.completions.create(
            model="mistral-7b-instruct-v0.1",
            messages=[
                {
                    "role": "system",
                    "content": f"Eres un técnico de soporte. Datos: {contexto}",
                },
                {"role": "user", "content": prompt},
            ],
        )
        full_response = response.choices[0].message.content
        st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
