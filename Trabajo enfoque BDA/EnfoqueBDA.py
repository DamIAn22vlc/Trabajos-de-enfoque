import os

# Suprimir warnings de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import time
import random
import sqlite3
from datetime import datetime
from confluent_kafka import Producer, Consumer
from typing import List
from fastapi import FastAPI
import pandas as pd
import tensorflow as tf

config = {"bootstrap.servers": "127.0.0.1:9092", "client.id": "simulador_smartmanutech"}
producer = Producer(config)


def generar_datos_sensores(maquina_id):
    """Genera lecturas coherentes de 5 sensores distintos."""

    # 1. Temperatura (C°) - Sube si hay mucha fricción o fallo
    temp = random.normalvariate(65, 2)

    # 2. Vibración (mm/s) - Aumenta drásticamente ante desgaste
    vibracion = random.uniform(0.1, 0.8)

    # 3. Velocidad (RPM) - Si baja de 1000, hay un cuello de botella
    rpm = random.randint(1150, 1250)

    # 4. Consumo Energético (kW) - Relacionado con el esfuerzo del motor
    consumo = (rpm / 500) + random.uniform(0.5, 1.2)

    # 5. Contador de Producción - Unidades totales y defectuosas
    unidades_ok = random.randint(5, 15)
    unidades_defecto = 1 if random.random() > 0.95 else 0  # 5% de probabilidad de fallo

    return {
        "ts": datetime.now().isoformat(),
        "mid": maquina_id,
        "sensores": {
            "termometro": round(temp, 2),
            "vibracion": round(vibracion, 4),
            "tacometro": rpm,
            "energia": round(consumo, 2),
            "contador": {"ok": unidades_ok, "bad": unidades_defecto},
        },
    }


try:
    print("🚀 Generando flujo de 5 sensores para SmartManuTech...")
    while True:
        # Simulamos 3 máquinas distintas trabajando en paralelo
        for m_id in ["MAQ_PRENSA_01", "MAQ_SOLDA_02", "MAQ_EMPAQUE_03"]:
            payload = generar_datos_sensores(m_id)

            producer.produce(
                "telemetria_industrial", key=m_id, value=json.dumps(payload)
            )

        producer.flush()
        print(f"Lote enviado a las {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(2)  # Frecuencia de muestreo

except KeyboardInterrupt:
    print("Simulación finalizada.")

    # Configuración del consumidor
config = {
    "bootstrap.servers": "127.0.0.1:9092",
    "group.id": "grupo_monitoreo_industrial",
    "auto.offset.reset": "latest",  # Solo leer lo nuevo que llegue
}
MODEL_PATH = "modelo_smartmanutech.h5"


def entrenar_modelo_inicial():
    """Crea y entrena un modelo rápido basado en los datos históricos existentes"""
    try:
        conn = sqlite3.connect("smartmanutech_historico.db")
        df = pd.read_sql_query(
            "SELECT temp, vibracion, rpm, consumo FROM telemetria", conn
        )
        conn.close()

        if len(df) < 10:
            print("⚠️ Datos insuficientes para entrenar IA. Usando modelo base...")
            return None

        # Etiquetado sintético para el entrenamiento inicial:
        # Riesgo (1) si temp > 67 o vibración > 0.7
        y = ((df["temp"] > 67) | (df["vibracion"] > 0.7)).astype(int)
        X = df[["temp", "vibracion", "rpm", "consumo"]]

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        model.fit(X, y, epochs=10, verbose=0)
        model.save(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Error en entrenamiento: {e}")
        return None


consumer = Consumer(config)
consumer.subscribe(["telemetria_industrial"])

print("🔍 Monitor de SmartManuTech activo... Escuchando sensores.")

try:
    while True:
        msg = consumer.poll(1.0)  # Espera 1 segundo por mensajes

        if msg is None:
            continue
        if msg.error():
            print(f"Error: {msg.error()}")
            continue

        # Decodificar el JSON
        data = json.loads(msg.value().decode("utf-8"))

        m_id = data["mid"]
        temp = data["sensores"]["termometro"]
        vibracion = data["sensores"]["vibracion"]

        # --- LÓGICA DE ALERTA (Mantenimiento Predictivo) ---
        if temp > 68.0:
            print(f"⚠️ [ALERTA] {m_id} - SOBRECALENTAMIENTO: {temp}°C")

        if vibracion > 0.75:
            print(
                f"🚨 [CRÍTICO] {m_id} - VIBRACIÓN EXCESIVA: {vibracion} mm/s. ¡Revisar rodamientos!"
            )

        # --- LÓGICA DE PRODUCCIÓN ---
        if data["sensores"]["contador"]["bad"] > 0:
            print(f"❌ [CALIDAD] {m_id} ha generado una unidad defectuosa.")

except KeyboardInterrupt:
    print("Monitor detenido.")
finally:
    consumer.close()

    # 1. Configuración de la Base de Datos SQLite
conn = sqlite3.connect("smartmanutech_historico.db")
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS telemetria (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mid TEXT,
        ts TEXT,
        temp REAL,
        vibracion REAL,
        rpm INTEGER,
        consumo REAL
    )
"""
)
conn.commit()

# 2. Configuración de Kafka
consumer = Consumer(
    {
        "bootstrap.servers": "127.0.0.1:9092",
        "group.id": "grupo_persistencia_sqlite",
        "auto.offset.reset": "earliest",
    }
)
consumer.subscribe(["telemetria_industrial"])

print("💾 Guardando telemetría en SQLite (smartmanutech_historico.db)...")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            continue

        data = json.loads(msg.value().decode("utf-8"))

        # 3. Insertar en SQLite
        cursor.execute(
            """
            INSERT INTO telemetria (mid, ts, temp, vibracion, rpm, consumo)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                data["mid"],
                data["ts"],
                data["sensores"]["termometro"],
                data["sensores"]["vibracion"],
                data["sensores"]["tacometro"],
                data["sensores"]["energia"],
            ),
        )
        conn.commit()
        print(f"✅ Guardado en DB: {data['mid']} - {data['sensores']['termometro']}°C")

except KeyboardInterrupt:
    print("Deteniendo persistencia...")
finally:
    conn.close()
    consumer.close()

app = FastAPI(title="SmartManuTech API 🏭")


def get_db_connection():
    conn = sqlite3.connect("smartmanutech_historico.db")
    conn.row_factory = sqlite3.Row  # Para que devuelva diccionarios
    return conn


@app.get("/")
def home():
    return {"status": "Online", "msg": "Bienvenido al Centro de Datos de SmartManuTech"}


@app.get("/telemetria")
def obtener_todo():
    """Devuelve los últimos 50 registros de la fábrica"""
    conn = get_db_connection()
    datos = conn.execute(
        "SELECT * FROM telemetria ORDER BY ts DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return [dict(row) for row in datos]


@app.get("/maquina/{machine_id}")
def filtrar_maquina(machine_id: str):
    """Filtra datos por una máquina específica"""
    conn = get_db_connection()
    query = "SELECT * FROM telemetria WHERE mid = ? ORDER BY ts DESC LIMIT 20"
    datos = conn.execute(query, (machine_id,)).fetchall()
    conn.close()
    return [dict(row) for row in datos]
