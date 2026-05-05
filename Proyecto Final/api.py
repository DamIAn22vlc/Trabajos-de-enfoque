"""
API REST para el Sistema de Gestión de Diabetes
Endpoints disponibles:
- POST /predict -> Predice la dosis de insulina
- GET /report -> Genera informe clínico
- GET /trends -> Verifica tendencias peligrosas
"""

from flask import Flask, request, jsonify, send_file
import torch
import os
from dia import (
    model,
    scaler_X,
    guardar_consulta,
    generar_informe_clinico,
    verificar_tendencia_peligrosa,
    procesar_mensaje_chatbot,
    init_db,
)

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))

# Inicializar base de datos al arrancar
init_db()


@app.route("/")
def index():
    """Sirve la página principal HTML"""
    return send_file("index.html")


@app.route("/predict", methods=["POST"])
def predict_bolus():
    """
    Predice la dosis de insulina basada en glucosa y carbohidratos

    Request JSON:
    {
        "glucosa": 154.0,
        "carbs": 50.0
    }
    """
    try:
        data = request.json
        glucosa = float(data.get("glucosa"))
        carbs = float(data.get("carbs"))

        # Validación de rango
        if glucosa < 40 or glucosa > 400:
            return (
                jsonify({"error": "Glucosa fuera de rango (40-400)"}),
                400,
            )
        if carbs < 0 or carbs > 200:
            return jsonify({"error": "Carbohidratos fuera de rango (0-200)"}), 400

        # Inferencia del modelo
        model.eval()
        with torch.no_grad():
            entrada = scaler_X.transform([[glucosa, carbs]])
            prediccion = model(torch.FloatTensor(entrada)).item()

        dosis = round(max(prediccion, 0), 2)  # Evitar dosis negativas

        # Guardar en BD
        guardar_consulta(glucosa, carbs, dosis)

        # Safety check
        advertencia = None
        if dosis > 15:
            advertencia = "⚠️ Dosis muy alta. Consulta con tu médico."

        return jsonify(
            {
                "status": "success",
                "glucosa": glucosa,
                "carbs": carbs,
                "dosis_recomendada": dosis,
                "unidad": "Unidades de Insulina",
                "advertencia": advertencia,
                "mensaje": f"Para {glucosa} mg/dL y {carbs}g de HC, se sugieren {dosis} U de insulina rápida.",
            }
        )

    except ValueError:
        return (
            jsonify(
                {
                    "error": "Datos inválidos. Asegúrate de enviar glucosa y carbs como números"
                }
            ),
            400,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chatbot", methods=["POST"])
def chatbot():
    """
    Procesa un mensaje en lenguaje natural y devuelve la recomendación

    Request JSON:
    {
        "mensaje": "Tengo 154 de glucosa y voy a comer 50g"
    }
    """
    try:
        data = request.json
        mensaje = data.get("mensaje", "")

        respuesta = procesar_mensaje_chatbot(mensaje)

        return jsonify({"status": "success", "respuesta": respuesta})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/report", methods=["GET"])
def report():
    """Genera un informe clínico con estadísticas del historial"""
    try:
        informe = generar_informe_clinico()

        return jsonify(
            {
                "status": "success",
                "informe": informe,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/trends", methods=["GET"])
def trends():
    """Verifica si hay tendencias peligrosas en los últimos datos"""
    try:
        tendencia = verificar_tendencia_peligrosa()

        if tendencia is None:
            return (
                jsonify(
                    {
                        "status": "warning",
                        "mensaje": "No hay suficientes datos para evaluar tendencias",
                    }
                ),
                200,
            )

        return jsonify({"status": "success", "tendencia": tendencia})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Verifica el estado de la API"""
    return jsonify({"status": "API funcionando correctamente", "version": "1.0"})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 SERVIDOR API - SISTEMA DE GESTIÓN DE DIABETES")
    print("=" * 60)
    print("\n📍 API disponible en: http://localhost:5000")
    print("\n📚 Endpoints disponibles:")
    print("  POST   /predict      -> Calcula dosis de insulina")
    print("  POST   /chatbot      -> Procesa mensaje en lenguaje natural")
    print("  GET    /report       -> Genera informe clínico")
    print("  GET    /trends       -> Verifica tendencias peligrosas")
    print("  GET    /health       -> Estado de la API")
    print("\n💡 Ejemplo de uso (con curl):")
    print("  curl -X POST http://localhost:5000/predict \\")
    print('    -H "Content-Type: application/json" \\')
    print('    -d "{\\"glucosa\\": 154, \\"carbs\\": 50}"')
    print("\n⛔ Presiona Ctrl+C para detener el servidor")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
