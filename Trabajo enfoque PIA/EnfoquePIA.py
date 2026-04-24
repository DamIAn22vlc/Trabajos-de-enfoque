import os
import boto3
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
BUCKET_NAME = os.getenv(
    "rekognition-custom-projects-us-east-1-55159ae1a8",
    "rekognition-custom-projects-us-east-1-55159ae1a8",
)

# Credenciales temporales
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", "ASIAUQRVZWAJXUEM5I4I")
SECRET_KEY = os.getenv("AWS_SECRET_KEY", "6h+3nsbXkuaKDGcYOwRrGjiZX57MeyuiSTwTfYSc")
SESSION_TOKEN = os.getenv(
    "AWS_SESSION_TOKEN",
    "IQoJb3JpZ2luX2VjELn//////////wEaCXVzLXdlc3QtMiJIMEYCIQDgc9lwjGEqHLb8ixbfBN5BuyDFzfWswh/SR7mLncTcFQIhAIvC60odJV+eTRrMaHG50xT1D9fQTZzREGLByxXAErg1KrgCCIL//////////wEQABoMMzEwNDI0MjE5NjY3IgxM0oVGWa67Gn4+oxMqjAKk9U78clOrUJzOLIVP39tmch/izs9dzqHPkqADe8ssK+XIPkno3JkiuJg0C8f8fLt6nVfTyY1QvUSL1f6sCk0tOzn0J0gsWrxVvMPpvxrMsmuGmIYCFaKfcEM8uQqcbjhcBegmezJ76JScutE/dbD4vOObmm8YjMEk3eKNxC4bIiFKTK4NbFdSyj8UKUgBcae0z5/74F2Tn4wcP3NSF5OJ9JHoFKyzYDU3FIu3QDQGdraqoxl4AqL6xr0qR3iqi+eK6eQsLQPAVjU4f4SX4Tt+FnUag6xYgiQUQ8b30wIVh1hmYY/+PJsCNgKvRR1tn9Q8woATvtAtcQ8N7fFnMbOmBn9gj79DYPCufMxVMK3Jrs8GOpwBDZvkvwjkp2YjBV/1JJl3QwvwNpXFpLWvCZn9cv2X4FMTVNJ64qjKaBm0uD1IrTh9BXXEsXDF/eKGggfMQVeIQSx32DeFjpzAODrffrsjTRt8F6W7df/dGT5QBL2MT8JlDdNeFti84PB7VJmvhRhuIXddubpTvjU6n5NkJJU4wImzE6ovKmpcETedXKmfiuT6MbZKp+Kz/7iDy4Y7",
)

# Crear la sesión usando el Token de Vocareum
try:
    session = boto3.Session(
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        aws_session_token=SESSION_TOKEN,
        region_name="us-east-1",
    )
    s3 = session.client("s3")
    rekognition = session.client("rekognition")
    logger.info("✓ Conexión a AWS iniciada correctamente")
except Exception as e:

    logger.error(f"✗ Error al conectar a AWS: {str(e)}")
    s3 = None
    rekognition = None

# Crear aplicación FastAPI
app = FastAPI(title="FastRetail AI", version="1.0")

# Configurar CORS para permitir acceso desde el HTML
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    """Servir el archivo HTML principal"""
    try:
        return FileResponse("index.html")
    except FileNotFoundError:
        return {
            "error": "index.html no encontrado. Asegurate que esté en el mismo directorio."
        }


@app.get("/health")
async def health():
    """Endpoint para verificar que el servidor está activo"""
    return {
        "status": "ok",
        "service": "FastRetail AI",
        "aws_configured": s3 is not None and rekognition is not None,
        "bucket": BUCKET_NAME,
    }


@app.post("/upload-and-analyze/")
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint para subir una imagen y analizarla con Amazon Rekognition
    """
    if s3 is None or rekognition is None:
        return {
            "error": "Servicio de AWS no configurado correctamente. Verifica tus credenciales."
        }

    file_name = file.filename
    logger.info(f"📤 Analizando archivo: {file_name}")

    try:
        # 1. Subir el archivo a Amazon S3
        logger.info(f"📦 Subiendo a S3 bucket: {BUCKET_NAME}")
        s3.upload_fileobj(file.file, BUCKET_NAME, file_name)
        logger.info(f"✓ Archivo subido a S3: {file_name}")
    except Exception as e:
        error_msg = f"Fallo al subir a S3: {str(e)}"
        logger.error(f"✗ {error_msg}")
        return {"error": error_msg}

    # 2. Llamar a Amazon Rekognition
    try:
        logger.info("🔍 Analizando con Rekognition...")
        response = rekognition.detect_labels(
            Image={"S3Object": {"Bucket": BUCKET_NAME, "Name": file_name}},
            MaxLabels=10,
            MinConfidence=80,  # Umbral de confianza configurado
        )

        # 3. Formatear resultados para el equipo de gestión
        etiquetas = []
        for label in response["Labels"]:
            etiquetas.append(
                {"nombre": label["Name"], "confianza": f"{label['Confidence']:.2f}%"}
            )

        logger.info(f"✓ Análisis completado: {len(etiquetas)} etiquetas detectadas")
        return {"archivo": file_name, "analisis": etiquetas, "status": "success"}

    except Exception as e:
        error_msg = f"Error en Rekognition: {str(e)}"
        logger.error(f"✗ {error_msg}")
        return {"error": error_msg}


async def pipeline_fast_retail(file):
    # PASO 1 & 2: Recepción y Validación
    filename = f"prod_{uuid.uuid4()}.jpg"

    # PASO 3: S3
    s3_client.upload_fileobj(file, BUCKET_NAME, filename)

    # PASO 4: Rekognition con Umbral de Confianza
    raw_labels = rekognition.detect_labels(
        Image={"S3Object": {"Bucket": BUCKET_NAME, "Name": filename}}, MinConfidence=85
    )

    # PASO 5: Guardar en DB (Simulado)
    db_record = {
        "image_url": f"https://{BUCKET_NAME}.s3.amazonaws.com/{filename}",
        "tags": [l["Name"] for l in raw_labels["Labels"]],
    }
    save_to_db(db_record)

    return db_record


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
