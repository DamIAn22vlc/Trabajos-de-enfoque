import os

# Suprimir warnings de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import sqlite3
import numpy as np
from datetime import datetime
import tensorflow as tf
import pandas as pd


# Variable global para el modelo
modelo_global = None


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

        # Etiquetado sintético: Riesgo (1) si temp > 67 o vibración > 0.7
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
        model.save("modelo_smartmanutech.h5")
        return model
    except Exception as e:
        print(f"Error en entrenamiento: {e}")
        return None


class SmartManuAPI:
    def __init__(self):
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup: Entrenar modelo al iniciar
            global modelo_global
            modelo_global = entrenar_modelo_inicial()
            yield
            # Shutdown: Limpieza si es necesaria

        self.app = FastAPI(title="SmartManuTech AI Edition 🤖🏭", lifespan=lifespan)
        self.model = None
        self.setup_routes()

    def setup_routes(self):

        @self.app.get("/")
        def home():
            global modelo_global
            return {
                "status": "Online",
                "ai_status": "Active" if modelo_global else "Collecting Data",
                "msg": "Centro de Datos con Mantenimiento Predictivo TensorFlow",
            }

        @self.app.get("/telemetria")
        def obtener_todo():
            conn = self.get_db_connection()
            datos = conn.execute(
                "SELECT * FROM telemetria ORDER BY ts DESC LIMIT 50"
            ).fetchall()
            conn.close()
            return [dict(row) for row in datos]

        @self.app.get("/prediccion/{machine_id}")
        def predecir_estado(machine_id: str):
            """Analiza la última lectura con TensorFlow para predecir fallos"""
            global modelo_global
            if not modelo_global:
                raise HTTPException(
                    status_code=503,
                    detail="Modelo de IA en entrenamiento o datos insuficientes",
                )

            conn = self.get_db_connection()
            last_row = conn.execute(
                "SELECT temp, vibracion, rpm, consumo FROM telemetria WHERE mid = ? ORDER BY ts DESC LIMIT 1",
                (machine_id,),
            ).fetchone()
            conn.close()

            if not last_row:
                raise HTTPException(
                    status_code=404, detail="No hay datos para esta máquina"
                )

            # Preparar datos para TensorFlow
            input_data = np.array(
                [
                    [
                        last_row["temp"],
                        last_row["vibracion"],
                        last_row["rpm"],
                        last_row["consumo"],
                    ]
                ]
            )
            probabilidad = float(modelo_global.predict(input_data, verbose=0)[0][0])

            estado = (
                "CRÍTICO"
                if probabilidad > 0.8
                else "PRECAUCIÓN" if probabilidad > 0.5 else "OPTIMO"
            )

            return {
                "machine_id": machine_id,
                "timestamp": datetime.now().isoformat(),
                "score_ia": round(probabilidad, 4),
                "estado_predictivo": estado,
                "probabilidad_fallo": f"{probabilidad * 100:.2f}%",
                "acciones": (
                    "Solicitar revisión técnica inmediata"
                    if estado == "CRÍTICO"
                    else "Mantener monitoreo"
                ),
            }

    def get_db_connection(self):
        conn = sqlite3.connect("smartmanutech_historico.db")
        conn.row_factory = sqlite3.Row
        return conn


# Instanciar la app para Uvicorn
api = SmartManuAPI()
app = api.app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
