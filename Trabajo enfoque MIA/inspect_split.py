import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

data = {
    "texto": [
        "señal debil",
        "la señal wifi se cae",
        "el wifi no llega al patio",
        "no hay señal en la cocina",
        "la wifi se desconecta frecuentemente",
        "internet lento",
        "la navegación está muy lenta",
        "la velocidad de internet es baja",
        "las descargas tardan mucho",
        "la página tarda en cargar",
        "cambiar clave",
        "no recuerdo la contraseña del wifi",
        "quiero resetear el password",
        "necesito cambiar la contraseña del router",
        "configurar wifi 5ghz",
        "luz roja router",
        "el router muestra error",
        "no hay conexión de fibra",
        "mi fibra óptica no funciona",
        "la linea de fibra está caída",
    ],
    "etiqueta": [
        "WIFI_COBERTURA",
        "WIFI_COBERTURA",
        "WIFI_COBERTURA",
        "WIFI_COBERTURA",
        "WIFI_COBERTURA",
        "WIFI_VELOCIDAD",
        "WIFI_VELOCIDAD",
        "WIFI_VELOCIDAD",
        "WIFI_VELOCIDAD",
        "WIFI_VELOCIDAD",
        "WIFI_CONFIG",
        "WIFI_CONFIG",
        "WIFI_CONFIG",
        "WIFI_CONFIG",
        "WIFI_CONFIG",
        "FIBRA_AVERIA",
        "FIBRA_AVERIA",
        "FIBRA_AVERIA",
        "FIBRA_AVERIA",
        "FIBRA_AVERIA",
    ],
}
df = pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(
    df["texto"], df["etiqueta"], test_size=0.4, random_state=42, stratify=df["etiqueta"]
)
print("lengths", len(X_train), len(X_test), len(y_train), len(y_test))
print("y_test counts")
print(y_test.value_counts())
modelo = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
    OneVsRestClassifier(
        LogisticRegression(max_iter=2000, random_state=42, solver="liblinear")
    ),
)
modelo.fit(X_train, y_train)
print("classes", modelo.named_steps["logisticregression"].classes_)
y_pred = modelo.predict(X_test)
print("y_pred counts")
print(pd.Series(y_pred).value_counts())
print("y_test", list(y_test))
print("y_pred", list(y_pred))
print("accuracy", sum(y_pred == y_test) / len(y_test))
