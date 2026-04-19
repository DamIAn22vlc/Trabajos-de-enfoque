import textwrap
from pathlib import Path

content = textwrap.dedent(
    r"""
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
                {"role": "system", "content": "Eres el asistente de soporte técnico de mi proyecto."},
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
        if isinstance(item, dict) and 'pregunta' in item:
            contexto += f"- Pregunta: {item['pregunta']}\n  Respuesta: {item['respuesta']}\n"
        elif isinstance(item, dict) and 'preguntas' in item:
            for sub in item['preguntas']:
                contexto += f"- Pregunta: {sub['pregunta']}\n  Respuesta: {sub['respuesta']}\n"

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


def cargar_modelo_entrenado(path='modelo_soporte_tecnico.pkl'):
    return joblib.load(path)


def predecir_categoria(prompt, modelo_path='modelo_soporte_tecnico.pkl'):
    modelo = cargar_modelo_entrenado(modelo_path)
    return modelo.predict([prompt])[0]


if __name__ == "__main__":
    test_asistente()
    entrenar_modelo()
"""
)
Path("EnfoqueMIA.py").write_text(content, encoding="utf-8")
print("EnfoqueMIA.py overwritten successfully")
