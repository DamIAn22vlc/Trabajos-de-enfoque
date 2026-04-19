import os
import json
from openai import OpenAI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Configuración para LM Studio
api_key = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key, base_url="http://127.0.0.1:1234/v1")


def test_asistente():
    try:
        completion = client.chat.completions.create(
            model="mistral-7b-instruct-v0.1",
            messages=[
                {
                    "role": "system",
                    "content": "Eres el asistente de soporte técnico de mi proyecto.",
                },
                {"role": "user", "content": "Hola, ¿puedes ayudarme con mi conexión?"},
            ],
        )
        print("Respuesta de la IA local:")
        print(completion.choices[0].message.content)
    except Exception as e:
        print(f"Error: Asegúrate de que el servidor esté en 'Started'. {e}")


def cargar_base_conocimiento():
    filename = 'datos.json' if os.path.exists('datos.json') else 'Data.json'
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def asistente_soporte(pregunta_usuario):
    datos = cargar_base_conocimiento()
    contexto = "Sigue estas instrucciones técnicas oficiales:\n"
    for item in datos:
        contexto += f"- Pregunta: {item['pregunta']}\n  Respuesta: {item['respuesta']}\n"

    completion = client.chat.completions.create(
        model="mistral-7b-instruct-v0.1",
        messages=[
            {
                "role": "system",
                "content": f"Eres un experto en soporte técnico. Responde de forma amable usando solo esta información: {contexto}"
            },
            {"role": "user", "content": pregunta_usuario}
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content


def entrenar_modelo():
    data = {
        'texto': [
            'señal debil',
            'la señal wifi se cae',
            'el wifi no llega al patio',
            'no hay señal en la cocina',
            'la wifi se desconecta frecuentemente',
            'internet lento',
            'la navegación está muy lenta',
            'la velocidad de internet es baja',
            'las descargas tardan mucho',
            'la página tarda en cargar',
            'cambiar clave',
            'no recuerdo la contraseña del wifi',
            'quiero resetear el password',
            'necesito cambiar la contraseña del router',
            'configurar wifi 5ghz',
            'luz roja router',
            'el router muestra error',
            'no hay conexión de fibra',
            'mi fibra óptica no funciona',
            'la linea de fibra está caída'
        ],
        'etiqueta': [
            'WIFI_COBERTURA',
            'WIFI_COBERTURA',
            'WIFI_COBERTURA',
            'WIFI_COBERTURA',
            'WIFI_COBERTURA',
            'WIFI_VELOCIDAD',
            'WIFI_VELOCIDAD',
            'WIFI_VELOCIDAD',
            'WIFI_VELOCIDAD',
            'WIFI_VELOCIDAD',
            'WIFI_CONFIG',
            'WIFI_CONFIG',
            'WIFI_CONFIG',
            'WIFI_CONFIG',
            'WIFI_CONFIG',
            'FIBRA_AVERIA',
            'FIBRA_AVERIA',
            'FIBRA_AVERIA',
            'FIBRA_AVERIA',
            'FIBRA_AVERIA',
            'FIBRA_AVERIA'
        ]
    }
    df = pd.DataFrame(data)

    X_train, X_test, y_train, y_test = train_test_split(
        df['texto'], df['etiqueta'], test_size=0.4, random_state=42, stratify=df['etiqueta']
    )

    modelo = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
        OneVsRestClassifier(LogisticRegression(max_iter=2000, random_state=42, solver='liblinear'))
    )
    modelo.fit(X_train, y_train)
    joblib.dump(modelo, 'modelo_soporte_tecnico.pkl')
    print("¡Modelo entrenado y guardado con éxito!")

    y_pred = modelo.predict(X_test)
    print(f"Precisión Global: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nReporte Detallado:")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    test_asistente()
    entrenar_modelo()


import streamlit as st
from openai import OpenAI
import joblib
import json

# Configuración de la página
st.set_page_config(page_title="Asistente Soporte Técnico", page_icon="🤖")
st.title("🤖 Soporte Técnico IA")
st.markdown("---")

# 1. Conexión con LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# 2. Cargar el Modelo Entrenado y el JSON
@st.cache_resource
def load_assets():
    modelo = joblib.load('modelo_soporte_tecnico.pkl')
    with open('datos.json', 'r', encoding='utf-8') as f:
        datos = json.load(f)
    return modelo, datos

modelo_entrenado, base_datos = load_assets()

# 3. Interfaz de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Lógica de la IA
    with st.chat_message("assistant"):
        # A) PREDICCIÓN CON EL MODELO ENTRENADO
        categoria = modelo_entrenado.predict([prompt])[0]
        st.caption(f"Categoría detectada: {categoria}")

        # B) BÚSQUEDA EN EL JSON (Simulada para el ejemplo)
        # Aquí podrías filtrar el JSON por la categoría predicha
        contexto = str(base_datos) 

        # C) GENERACIÓN CON LM STUDIO (Mistral)
        response = client.chat.completions.create(
            model="mistral-7b-instruct-v0.1",
            messages=[
                {"role": "system", "content": f"Eres un técnico de soporte. Datos: {contexto}"},
                {"role": "user", "content": prompt}
            ]
        )
        full_response = response.choices[0].message.content
        st.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})


    def es_seguro(texto_usuario):
    palabras_prohibidas = ["admin", "password", "root", "script"]
    return not any(palabra in texto_usuario.lower() for palabra in palabras_prohibidas)

