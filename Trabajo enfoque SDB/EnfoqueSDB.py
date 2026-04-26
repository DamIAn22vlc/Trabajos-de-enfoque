from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import json
import os

# Importaciones de Spark movidas arriba para mantener el estándar PEP8
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, when
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor

app = FastAPI(title="SuperFresh Inventory API - MongoDB Edition")


# 1. MODELO ALINEADO CON EL JSON DE 60 PRODUCTOS
class DatosPrediccion(BaseModel):
    sucursal: str = Field(..., description="Antes id_tienda")
    producto: str = Field(..., description="Antes id_producto")
    categoria: str
    precio: float
    dia_semana: int
    es_promocion: int
    temperatura: float


# 2. CONFIGURACIÓN EXACTA DE MONGODB
MONGO_DETAILS = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client.db  # Apuntando a tu base de datos real
collection = database.get_collection("SuperFresh")  # Apuntando a tu colección real


@app.post("/predict")
async def predecir_y_registrar(datos: DatosPrediccion):
    # Lógica del modelo de IA (Simulada)
    prediccion_unidades = 75 if datos.es_promocion else 40

    # Estructura del documento compatible con tu esquema de Power BI
    documento_prediccion = {
        "sucursal": datos.sucursal,
        "producto": datos.producto,
        "categoria": datos.categoria,
        "precio": datos.precio,
        "valor_predicho": prediccion_unidades,
        "metadata": {
            "temp_ambiente": datos.temperatura,
            "dia": datos.dia_semana,
            "es_promo": bool(datos.es_promocion),
        },
        "timestamp": datetime.now(timezone.utc),
        "tipo_registro": "prediccion_api",  # <-- Clave para filtrar en Power BI
    }

    # Guardar en MongoDB de forma asíncrona
    nuevo_registro = await collection.insert_one(documento_prediccion)

    return {
        "id": str(nuevo_registro.inserted_id),
        "prediccion": prediccion_unidades,
        "status": "stored_in_mongodb",
    }


@app.get("/historico/{nombre_producto}")
async def obtener_historico_predicciones(nombre_producto: str):
    """
    Recupera las últimas predicciones usando la nueva nomenclatura 'producto'.
    """
    cursor = collection.find({"producto": nombre_producto}).limit(10)
    resultados = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])  # Convertir ObjectId a String para JSON
        resultados.append(doc)
    return resultados


# --- INTEGRACIÓN CON APACHE SPARK ---


def create_spark_mongo_session():
    """
    Configura Spark para leer y escribir en la misma colección que Power BI y FastAPI.
    """
    return (
        SparkSession.builder.appName("SuperFresh_Mongo_Integration")
        .config(
            "spark.mongodb.read.connection.uri",
            "mongodb://localhost:27017/db.SuperFresh",
        )
        .config(
            "spark.mongodb.write.connection.uri",
            "mongodb://localhost:27017/db.SuperFresh",  # <-- Corregido
        )
        .config(
            "spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.1.1"
        )
        .getOrCreate()
    )


def save_predictions_to_mongo(predictions_df):
    """Guarda predicciones en MongoDB con configuración completa."""
    try:
        print("Guardando predicciones en MongoDB...")
        predictions_df.write.format("mongodb").option(
            "connection.uri", "mongodb://localhost:27017/db.SuperFresh"
        ).option("database", "db").option("collection", "SuperFresh").mode(
            "append"
        ).save()
        print("✅ Predicciones guardadas exitosamente en MongoDB.")
    except Exception as e:
        print(f"❌ Error al guardar en MongoDB: {str(e)}")
        raise


def evaluar_modelo(predictions):
    rmse_evaluador = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse"
    )
    r2_evaluador = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="r2"
    )
    mae_evaluador = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="mae"
    )

    rmse = rmse_evaluador.evaluate(predictions)
    r2 = r2_evaluador.evaluate(predictions)
    mae = mae_evaluador.evaluate(predictions)

    return {"RMSE": rmse, "R2": r2, "MAE": mae}


def pipeline_optimizado(df):
    # Reparticionamos por 'sucursal' (nomenclatura actualizada)
    df_reparticionado = df.repartition(col("sucursal"))
    df_reparticionado.cache()
    return df_reparticionado


async def setup_mongodb_optimization():
    # Índices adaptados a los campos reales del JSON para búsquedas ultra rápidas
    await collection.create_index([("sucursal", 1), ("producto", 1)])
    await collection.create_index("timestamp", expireAfterSeconds=2592000)
    print("✅ Índices de MongoDB optimizados.")


def hyperparameter_tuning(rf, train_data, evaluator):
    paramGrid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [50, 100, 200])
        .addGrid(rf.maxDepth, [10, 20])
        .addGrid(rf.featureSubsetStrategy, ["auto", "sqrt"])
        .build()
    )

    cv = CrossValidator(
        estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5
    )
    return cv.fit(train_data)


# --- VISUALIZACIÓN DE MÉTRICAS ---


def cargar_datos_json():
    """Carga los datos del archivo JSON limpio."""
    # Intentar cargar datos_limpios.json primero
    ruta_json = os.path.join(os.path.dirname(__file__), "datos_limpios.json")
    
    # Si no existe, intentar con productos.json
    if not os.path.exists(ruta_json):
        ruta_json = os.path.join(os.path.dirname(__file__), "productos.json")
    
    try:
        with open(ruta_json, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        print(f"✅ Cargados {len(datos)} registros de {os.path.basename(ruta_json)}")
        return datos
    except Exception as e:
        print(f"❌ Error al cargar JSON: {e}")
        return []


def entrenar_modelo_con_datos_reales():
    """Entrena un modelo Random Forest con los datos reales del JSON."""
    try:
        # Cargar datos
        spark = create_spark_mongo_session()
        datos = cargar_datos_json()
        
        if not datos:
            return None, None
        
        # Crear DataFrame de Spark
        df = spark.createDataFrame(datos)
        
        # Mapeo de categorías a números
        categoria_map = {
            "Lacteos": 1, "Panaderia": 2, "Fruteria": 3, "Limpieza": 4,
            "Carniceria": 5, "Bebidas": 6, "Despensa": 7
        }
        
        # Agregar columnas numéricas
        df_procesado = df \
            .withColumn("cat_num", when(col("categoria").isin(list(categoria_map.keys())), 
                        col("categoria")).otherwise("Despensa")) \
            .withColumn("pago_num", when(col("pago") == "Tarjeta", 0).otherwise(1)) \
            .withColumn("sucursal_num", when(col("sucursal") == "Norte", 0)
                        .when(col("sucursal") == "Sur", 1)
                        .when(col("sucursal") == "Centro", 2).otherwise(3)) \
            .withColumn("precio", col("precio").cast("double")) \
            .withColumn("cantidad", col("cantidad").cast("double"))
        
        # Features
        vector_assembler = VectorAssembler(
            inputCols=["precio", "pago_num", "sucursal_num"],
            outputCol="features"
        )
        df_features = vector_assembler.transform(df_procesado)
        
        # Dividir datos
        train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=42)
        
        # Entrenar modelo
        rf = RandomForestRegressor(
            labelCol="cantidad",
            featuresCol="features",
            numTrees=50,
            maxDepth=10,
            seed=42
        )
        model = rf.fit(train_data)
        
        # Hacer predicciones
        predictions = model.transform(test_data)
        
        # Calcular métricas
        metricas = evaluar_modelo(predictions)
        
        print("✅ Modelo entrenado con éxito")
        print(f"  RMSE: {metricas['RMSE']:.2f}")
        print(f"  R²: {metricas['R2']:.2f}")
        print(f"  MAE: {metricas['MAE']:.2f}")
        
        return metricas, predictions
    except Exception as e:
        print(f"❌ Error al entrenar modelo: {e}")
        return None, None


# Variables globales para almacenar métricas
metricas_globales = None


@app.on_event("startup")
async def startup_event():
    """Entrena el modelo al iniciar la API."""
    global metricas_globales
    metricas_globales, _ = entrenar_modelo_con_datos_reales()
    await setup_mongodb_optimization()


@app.get("/metricas")
async def visualizar_metricas():
    """
    Endpoint para visualizar las métricas de precisión del modelo en HTML interactivo.
    Lee datos reales de productos.json y los procesa con Spark.
    """
    global metricas_globales
    
    # Si no hay métricas, usar valores simulados
    if metricas_globales is None:
        metricas_ejemplo = {"RMSE": 12.45, "R2": 0.87, "MAE": 8.32}
        status = "⚠️ Usando datos simulados"
    else:
        metricas_ejemplo = metricas_globales
        status = "✅ Datos en tiempo real desde productos.json"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Métricas del Modelo - SuperFresh</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }}
            h1 {{
                text-align: center;
                color: #333;
                margin-bottom: 10px;
            }}
            .status {{
                text-align: center;
                font-size: 14px;
                color: #666;
                margin-bottom: 30px;
                padding: 10px;
                background: #f0f0f0;
                border-radius: 5px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}
            .metric-card h3 {{
                margin: 0 0 10px 0;
                font-size: 14px;
                text-transform: uppercase;
                opacity: 0.9;
            }}
            .metric-card .value {{
                font-size: 32px;
                font-weight: bold;
            }}
            .metric-card .description {{
                font-size: 12px;
                margin-top: 10px;
                opacity: 0.8;
            }}
            .charts-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 30px;
            }}
            .chart-container {{
                position: relative;
                height: 300px;
                background: #f9f9f9;
                border-radius: 8px;
                padding: 20px;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Métricas de Precisión del Modelo - SuperFresh</h1>
            <div class="status">{status}</div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>RMSE</h3>
                    <div class="value">{metricas_ejemplo["RMSE"]:.2f}</div>
                    <div class="description">Error Cuadrático Medio (menor es mejor)</div>
                </div>
                <div class="metric-card">
                    <h3>R² Score</h3>
                    <div class="value">{metricas_ejemplo["R2"]:.2f}</div>
                    <div class="description">Varianza Explicada (0-1, mayor es mejor)</div>
                </div>
                <div class="metric-card">
                    <h3>MAE</h3>
                    <div class="value">{metricas_ejemplo["MAE"]:.2f}</div>
                    <div class="description">Error Absoluto Medio (menor es mejor)</div>
                </div>
            </div>
            
            <div class="charts-grid">
                <div class="chart-container">
                    <canvas id="metricsChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="r2Chart"></canvas>
                </div>
            </div>
            
            <div class="footer">
                <p>SuperFresh Inventory API - Modelo de Predicción en Tiempo Real</p>
                <p>Última actualización: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            </div>
        </div>
        
        <script>
            const metricas = {metricas_ejemplo};
            
            // Gráfico de Métricas
            const ctxMetrics = document.getElementById('metricsChart').getContext('2d');
            new Chart(ctxMetrics, {{
                type: 'bar',
                data: {{
                    labels: ['RMSE', 'MAE'],
                    datasets: [{{
                        label: 'Errores (menor es mejor)',
                        data: [metricas.RMSE, metricas.MAE],
                        backgroundColor: ['#FF6384', '#36A2EB'],
                        borderRadius: 5,
                        borderSkipped: false
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{display: true, position: 'top'}}
                    }},
                    scales: {{
                        y: {{beginAtZero: true}}
                    }}
                }}
            }});
            
            // Gráfico R² Score
            const ctxR2 = document.getElementById('r2Chart').getContext('2d');
            new Chart(ctxR2, {{
                type: 'doughnut',
                data: {{
                    labels: ['Varianza Explicada', 'Varianza No Explicada'],
                    datasets: [{{
                        data: [metricas.R2 * 100, (1 - metricas.R2) * 100],
                        backgroundColor: ['#4BC0C0', '#FFCE56']
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{display: true, position: 'bottom'}}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)
