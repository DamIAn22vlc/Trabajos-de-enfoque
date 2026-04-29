import sqlite3

# Crear la base de datos SQLite
conn = sqlite3.connect("smartmanutech_historico.db")
cursor = conn.cursor()

# Crear tabla de telemetría
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
print("✅ Base de datos creada: smartmanutech_historico.db")
print("✅ Tabla 'telemetria' creada exitosamente")

conn.close()
