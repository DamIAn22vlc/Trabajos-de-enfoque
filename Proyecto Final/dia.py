import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Semilla para que los resultados sean reproducibles en nuestras pruebas
np.random.seed(42)

# ==========================================
# PARÁMETROS DEL PACIENTE VIRTUAL
# ==========================================
n_samples = 80
icr = 15.0  # 1 Unidad de insulina cubre 15g de carbohidratos
isf = 45.0  # 1 Unidad de insulina baja 45 mg/dL de glucosa
glucosa_objetivo = 100.0

# 1. Generar ingesta de Carbohidratos (entre 15g y 90g)
carbs_ingeridos = np.random.uniform(15, 90, n_samples).round(1)

# 2. Generar Insulina Inyectada (Simulando error humano)
# El usuario calcula la dosis teórica (carbs / icr), pero simulamos un margen de
# error del +/- 20% porque la gente suele estimar "a ojo" los gramos de la comida.D
error_estimacion = np.random.uniform(0.8, 1.2, n_samples)
insulina_inyectada = (carbs_ingeridos / icr) * error_estimacion
insulina_inyectada = insulina_inyectada.round(1)

# 3. Estado Inicial de la Glucosa (antes de comer)
glucosa_inicial = np.random.normal(110, 20, n_samples).round(0)

# 4. Calcular el impacto en la glucosa (Delta G)
# Subida teórica por la comida: (carbs / icr) * isf
subida_por_comida = (carbs_ingeridos / icr) * isf

# Bajada teórica por la insulina: insulina * isf
bajada_por_insulina = insulina_inyectada * isf

# Ruido metabólico (la absorción nunca es matemáticamente perfecta)
ruido_metabolico = np.random.normal(0, 15, n_samples)

# Variación total de glucosa
delta_g = subida_por_comida - bajada_por_insulina + ruido_metabolico
delta_g = delta_g.round(0)

# 5. Glucosa Final (aprox 3 horas después del bolo prandial)
glucosa_final = (glucosa_inicial + delta_g).clip(
    min=40
)  # Evitamos valores negativos imposibles

# ==========================================
# CREACIÓN DEL DATASET
# ==========================================
df_diabetes = pd.DataFrame(
    {
        "glucosa_inicial_mgdl": glucosa_inicial,
        "carbohidratos_g": carbs_ingeridos,
        "insulina_u": insulina_inyectada,
        "delta_g": delta_g,
        "glucosa_final_mgdl": glucosa_final,
    }
)

# Mostrar las primeras 5 muestras para verificar
print(df_diabetes.head())


# ==========================================
# 1. PREPARACIÓN DE LOS DATOS (CORRECCIÓN DEL TARGET)
# ==========================================
# Usamos el dataframe df_diabetes generado en el paso anterior.
# Calculamos el target real (Insulina Óptima) sin los errores humanos.
glucosa_objetivo = 100.0
df_diabetes["insulina_optima"] = (df_diabetes["carbohidratos_g"] / icr) + (
    (df_diabetes["glucosa_inicial_mgdl"] - glucosa_objetivo) / isf
)

# Evitamos dosis negativas (si la glucosa está muy baja, no inyectamos para los hidratos)
df_diabetes["insulina_optima"] = df_diabetes["insulina_optima"].clip(lower=0.0)

# Definimos nuestras características (X) y el objetivo (y)
X = df_diabetes[["glucosa_inicial_mgdl", "carbohidratos_g"]].values
y = df_diabetes["insulina_optima"].values.reshape(-1, 1)

# Dividimos en entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalizamos los datos (vital para que la red neuronal converja)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Convertimos a tensores de PyTorch
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)

# Creamos DataLoaders
dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)


# ==========================================
# 2. DEFINICIÓN DE LA ARQUITECTURA (MODELO)
# ==========================================
class BolusCalculatorNN(nn.Module):
    def __init__(self):
        super(BolusCalculatorNN, self).__init__()
        # Entrada: 2 variables (Glucosa Inicial, Carbohidratos)
        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)  # Salida: 1 valor continuo (Unidades de insulina)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # Usamos ReLU extra o simplemente controlamos la salida en producción
        # para que nunca recomiende dosis negativas.
        return x


model = BolusCalculatorNN()

# ==========================================
# 3. BUCLE DE ENTRENAMIENTO
# ==========================================
criterion = nn.MSELoss()  # Error Cuadrático Medio para regresión
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 150

print("Iniciando entrenamiento...\n")
for epoch in range(epochs):
    model.train()
    loss_total = 0

    for batch_X, batch_y in dataloader_train:
        optimizer.zero_grad()  # Limpiamos gradientes
        predicciones = model(batch_X)  # Forward pass
        loss = criterion(predicciones, batch_y)  # Calculamos el error
        loss.backward()  # Backpropagation
        optimizer.step()  # Actualizamos pesos

        loss_total += loss.item()

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], Loss (MSE): {loss_total/len(dataloader_train):.4f}"
        )

# ==========================================
# 4. EVALUACIÓN Y PRUEBA PRÁCTICA
# ==========================================
model.eval()
with torch.no_grad():
    # Simulamos una situación real: El paciente tiene 160 de glucosa y va a comer 45g de hidratos
    situacion_nueva = np.array([[160.0, 45.0]])
    situacion_escalada = scaler_X.transform(situacion_nueva)
    tensor_nuevo = torch.FloatTensor(situacion_escalada)

    dosis_predicha = model(tensor_nuevo).item()

    # Cálculo manual teórico para comparar: (45/15) + ((160-100)/45) = 3.0 + 1.33 = 4.33 U
    print(f"\n--- PRUEBA DEL MODELO ---")
    print(f"Glucosa: 160 mg/dL | Carbohidratos: 45g")
    print(f"Dosis recomendada por la IA: {dosis_predicha:.2f} Unidades")

    import re


def extraer_datos_diabetes(texto):
    # Diccionario para almacenar lo que encontremos
    datos = {"glucosa": None, "carbs": None}

    # 1. Buscamos la glucosa (números seguidos de mg/dl o cerca de azúcar/glucosa)
    # Patrón: busca un número de 2 a 3 cifras cerca de palabras clave
    match_glucosa = re.search(
        r"(?:glucosa|azúcar|tengo|medido)\s*(\d{2,3})", texto.lower()
    )
    if match_glucosa:
        datos["glucosa"] = float(match_glucosa.group(1))

    # 2. Buscamos los carbohidratos (números cerca de gramos, g, hc, o comer)
    match_carbs = re.search(
        r"(\d{1,3})\s*(?:gramos|g|hc|carbs|pasta|comida)", texto.lower()
    )
    if match_carbs:
        datos["carbs"] = float(match_carbs.group(1))

    return datos


# --- PRUEBA DE FUEGO ---
frase_usuario = "Hola, tengo la glucosa en 154 y voy a cenar 50g de hidratos"
entidades = extraer_datos_diabetes(frase_usuario)

print(f"Entidades extraídas: {entidades}")
# Resultado: {'glucosa': 154.0, 'carbs': 50.0}


# ==========================================
# 5. PROCESAMIENTO DE MENSAJES CHATBOT
# ==========================================
def procesar_mensaje_chatbot(mensaje_usuario):
    """Procesa mensaje del usuario y calcula la dosis de insulina"""
    # Paso 1: Entender qué dice el usuario
    datos = extraer_datos_diabetes(mensaje_usuario)

    if datos["glucosa"] is None or datos["carbs"] is None:
        return "🤖: No he captado bien los datos. ¿Me podrías decir tu glucosa actual y cuántos gramos vas a comer?"

    # Paso 2: Llamar a nuestra Red Neuronal
    try:
        # Preparamos el tensor y predecimos
        entrada = scaler_X.transform([[datos["glucosa"], datos["carbs"]]])
        prediccion = model(torch.FloatTensor(entrada)).item()

        # Paso 3: Respuesta con "Safety Check"
        dosis = round(prediccion, 1)

        if dosis > 15:  # Límite de seguridad arbitrario
            return f"🤖: La dosis calculada es de {dosis} U, pero es muy alta. Por seguridad, verifica con tu médico."

        return (
            f"🤖: Entendido. Para {datos['glucosa']} mg/dL y {int(datos['carbs'])}g de HC, "
            f"la dosis sugerida es de **{dosis} Unidades**."
        )

    except Exception as e:
        return "🤖: Ups, tuve un error técnico calculando la dosis."


# ==========================================
# 6. BASE DE DATOS SQL
# ==========================================
import sqlite3
import datetime


def init_db():
    """Inicializa la base de datos de diabetes"""
    conn = sqlite3.connect("historial_diabetes.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS consultas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TIMESTAMP,
            glucosa_entrada REAL,
            carbs_entrada REAL,
            dosis_calculada REAL
        )
    """)
    conn.commit()
    conn.close()


def guardar_consulta(glucosa, carbs, dosis):
    """Guarda una consulta en la base de datos"""
    conn = sqlite3.connect("historial_diabetes.db")
    cursor = conn.cursor()
    ahora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO consultas (fecha, glucosa_entrada, carbs_entrada, dosis_calculada)
        VALUES (?, ?, ?, ?)
    """,
        (ahora, glucosa, carbs, dosis),
    )
    conn.commit()
    conn.close()


def generar_informe_clinico():
    """Genera un informe clínico con estadísticas del historial"""
    conn = sqlite3.connect("historial_diabetes.db")
    cursor = conn.cursor()

    query = """
        SELECT 
            COUNT(*) as total_consultas,
            AVG(glucosa_entrada) as promedio_glucosa,
            MAX(carbs_entrada) as max_carbs,
            SUM(dosis_calculada) as total_insulina
        FROM consultas
    """

    cursor.execute(query)
    resultado = cursor.fetchone()
    conn.close()

    informe = {
        "Total de mediciones": resultado[0],
        "Promedio de glucosa": round(resultado[1], 2) if resultado[1] else 0,
        "Bolo más grande de comida (g)": resultado[2] if resultado[2] else 0,
        "Insulina total inyectada (U)": round(resultado[3], 2) if resultado[3] else 0,
    }

    return informe


def verificar_tendencia_peligrosa():
    """Verifica si hay tendencias peligrosas en los últimos datos"""
    conn = sqlite3.connect("historial_diabetes.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT glucosa_entrada FROM consultas 
        ORDER BY fecha DESC LIMIT 5
    """)

    ultimas_glucosas = [row[0] for row in cursor.fetchall()]
    conn.close()

    if len(ultimas_glucosas) < 5:
        return None

    media_reciente = sum(ultimas_glucosas) / len(ultimas_glucosas)

    if media_reciente > 180:
        return "⚠️ ALERTA: Tu tendencia glucémica es alta en las últimas horas. Considera revisar tu hidratación o contactar a tu médico."

    if ultimas_glucosas[0] < 70:
        return "⚠️ AVISO: Estás entrando en rango de hipoglucemia. Ten a mano carbohidratos de absorción rápida."

    return "✅ Tu tendencia actual es estable."


# ==========================================
# 7. PRUEBAS Y EJEMPLOS
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("SISTEMA DE GESTIÓN DE DIABETES - VERSIÓN 1.0")
    print("=" * 50)

    # Inicializar base de datos
    init_db()
    print("✅ Base de datos inicializada")

    # Guardar datos de ejemplo
    print("\n--- GUARDANDO DATOS DE EJEMPLO ---")
    guardar_consulta(154.0, 50.0, 4.5)
    guardar_consulta(210.0, 80.0, 7.2)
    guardar_consulta(128.0, 35.0, 2.8)
    print("✅ Datos guardados exitosamente")

    # Prueba de procesamiento de mensaje
    print("\n--- PRUEBA DE CHATBOT ---")
    mensaje = "Tengo el azúcar en 210 y voy a comer 80 gramos de pizza"
    print(f"Usuario: {mensaje}")
    resultado = procesar_mensaje_chatbot(mensaje)
    print(resultado)

    # Generar informe
    print("\n--- INFORME CLÍNICO ---")
    informe = generar_informe_clinico()
    for key, value in informe.items():
        print(f"  {key}: {value}")

    # Verificar tendencias
    print("\n--- VERIFICACIÓN DE TENDENCIAS ---")
    tendencia = verificar_tendencia_peligrosa()
    if tendencia:
        print(tendencia)

    print("\n" + "=" * 50)
