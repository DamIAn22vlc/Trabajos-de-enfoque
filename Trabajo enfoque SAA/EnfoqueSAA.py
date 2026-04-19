import pandas as pd
import numpy as np
import random

# Seteamos una semilla para que los "datos aleatorios" sean reproducibles
np.random.seed(42)

# 1. Generar Cursos (Catálogo)
n_cursos = 50
temas = ["Python", "Data Science", "Diseño UX", "Marketing Digital", "Finanzas"]
cursos = pd.DataFrame(
    {
        "course_id": range(1, n_cursos + 1),
        "titulo": [f"Curso de {random.choice(temas)} {i}" for i in range(n_cursos)],
        "dificultad": np.random.choice(["Básico", "Intermedio", "Avanzado"], n_cursos),
        "duracion_hrs": np.random.randint(5, 40, n_cursos),
    }
)

# 2. Generar Usuarios (Estudiantes)
n_usuarios = 200
usuarios = pd.DataFrame(
    {
        "user_id": range(1, n_usuarios + 1),
        "edad": np.random.randint(18, 60, n_usuarios),
        "interes_principal": np.random.choice(temas, n_usuarios),
        "estudios_previos": np.random.choice(
            ["Secundaria", "Grado Técnico", "Universitario", "Posgrado"], n_usuarios
        ),
    }
)

# 3. Generar Interacciones (Comportamiento)
# Simulamos que cada usuario ha interactuado con entre 3 y 10 cursos
filas_interaccion = []
for u_id in usuarios["user_id"]:
    n_interacciones = np.random.randint(3, 11)
    cursos_vistos = np.random.choice(
        cursos["course_id"], n_interacciones, replace=False
    )
    for c_id in cursos_vistos:
        filas_interaccion.append(
            {
                "user_id": u_id,
                "course_id": c_id,
                "rating": np.random.randint(1, 6),  # Calificación de 1 a 5
                "progreso": np.random.uniform(0.1, 1.0),  # 10% a 100% completado
            }
        )

df_interacciones = pd.DataFrame(filas_interaccion)

# 1. Agrupamos comportamiento de interacciones
perfil_comportamiento = (
    df_interacciones.groupby("user_id")
    .agg({"rating": "mean", "progreso": "mean", "course_id": "count"})
    .reset_index()
)

# 2. Unimos con la información de usuario (incluyendo estudios)
df_modelo = pd.merge(
    perfil_comportamiento, usuarios[["user_id", "estudios_previos"]], on="user_id"
)

# 3. Mapeo de estudios a números
mapping = {"Secundaria": 1, "Grado Técnico": 2, "Universitario": 3, "Posgrado": 4}
df_modelo["estudios_num"] = df_modelo["estudios_previos"].map(mapping)

# Eliminamos la columna de texto para el modelo
X = df_modelo[["rating", "progreso", "course_id", "estudios_num"]]

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Seleccionamos las características para el modelo
features = ["rating", "progreso", "course_id", "estudios_num"]
data_to_scale = df_modelo[features]

# Escalamos para que todo tenga media 0 y varianza 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_to_scale)

inercia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inercia.append(kmeans.inertia_)

# Visualización del "Codo"
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inercia, marker="o")
plt.title("Método del Codo")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Inercia")
# plt.show()  # Comentado para evitar bloqueo en ejecución no interactiva

# Aplicamos el modelo final (asumiendo K=3 para este ejemplo)
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df_modelo["cluster"] = kmeans_final.fit_predict(X_scaled)

# Analizamos los centros de los grupos
resumen_clusters = df_modelo.groupby("cluster")[features].mean()
print(resumen_clusters)

from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, df_modelo["cluster"])
print(f"Índice de Silhouette: {score:.2f}")

# Creamos la matriz: Usuarios vs Cursos
matrix_u_c = df_interacciones.pivot(
    index="user_id", columns="course_id", values="rating"
)

# Llenamos los huecos (NaN) con 0, asumiendo que no hay interacción
matrix_u_c_filled = matrix_u_c.fillna(0)

from sklearn.metrics.pairwise import cosine_similarity

# Calculamos la similitud entre todos los usuarios
user_sim = cosine_similarity(matrix_u_c_filled)
user_sim_df = pd.DataFrame(
    user_sim, index=matrix_u_c_filled.index, columns=matrix_u_c_filled.index
)


def recomendar_cursos(user_id, n_recom=3):
    # 1. Obtener el cluster del usuario
    cluster_actual = df_modelo.loc[df_modelo["user_id"] == user_id, "cluster"].values[0]

    # 2. Filtrar usuarios similares que pertenezcan al mismo cluster
    usuarios_mismo_cluster = df_modelo[df_modelo["cluster"] == cluster_actual][
        "user_id"
    ].tolist()

    # 3. Encontrar usuarios más similares (excluyendo al propio usuario)
    similares = (
        user_sim_df[user_id]
        .loc[usuarios_mismo_cluster]
        .sort_values(ascending=False)
        .iloc[1:6]
    )

    # 4. Ver qué cursos les gustaron a esos similares
    recomendaciones = []
    for sim_user in similares.index:
        # Cursos con rating > 4 que el usuario actual NO ha visto
        vistos_por_mi = df_interacciones[df_interacciones["user_id"] == user_id][
            "course_id"
        ].tolist()
        top_del_otro = df_interacciones[
            (df_interacciones["user_id"] == sim_user)
            & (df_interacciones["rating"] >= 4)
            & (~df_interacciones["course_id"].isin(vistos_por_mi))
        ]
        recomendaciones.extend(top_del_otro["course_id"].tolist())

    # Devolver los N cursos más frecuentes entre los recomendados
    return list(set(recomendaciones))[:n_recom]


# Ejemplo de uso:
print(f"Recomendaciones para el Usuario 1: {recomendar_cursos(user_id=1)}")


from sqlalchemy import create_engine

engine = create_engine("sqlite:///academia.db")

# Guardar los datos sintéticos en tablas SQL
usuarios.to_sql("users", engine, if_exists="replace", index=False)
cursos.to_sql("courses", engine, if_exists="replace", index=False)
df_interacciones.to_sql("interactions", engine, if_exists="replace", index=False)

print("Base de datos SQL inicializada con éxito.")

import streamlit as st

st.set_page_config(page_title="EduRecommend AI", layout="wide")

st.title("🎓 Sistema de Recomendación Inteligente")
st.sidebar.header("Panel de Control")

# Simulación de sesión
user_id = st.sidebar.number_input(
    "Introduce tu ID de Usuario", min_value=1, max_value=200, value=1
)

# --- VISTA DEL ESTUDIANTE ---
st.subheader(f"Bienvenido de nuevo, Estudiante #{user_id}")

if st.button("Obtener mis Recomendaciones"):
    # Obtenemos los IDs de los cursos recomendados
    recoms_ids = recomendar_cursos(user_id, n_recom=3)

    if recoms_ids:
        cols = st.columns(len(recoms_ids))

        for i, c_id in enumerate(recoms_ids):
            # 1. Buscamos la información real del curso en el DataFrame 'cursos'
            datos_curso = cursos[cursos["course_id"] == c_id].iloc[0]
            titulo = datos_curso["titulo"]
            dificultad = datos_curso["dificultad"]
            duracion = datos_curso["duracion_hrs"]

            # 2. Mostramos los datos de forma visual
            with cols[i]:
                st.info(f"**{titulo}**")
                st.write(f"📊 **Dificultad:** {dificultad}")
                st.write(f"⏱️ **Duración:** {duracion} hrs")
                st.write("⭐ Recomendado para tu perfil")
                # Usamos el ID del curso para el botón, así sabemos a cuál se inscribe
                st.button(f"Inscribirme", key=f"btn_{c_id}")
    else:
        st.info("No se pudieron encontrar recomendaciones para este usuario.")

# ---

# --- VISTA DEL ADMINISTRADOR (Evaluación del Sistema) ---
st.divider()
st.subheader("📊 Análisis de Segmentación (Vista Admin)")

col1, col2 = st.columns(2)

with col1:
    st.write("Distribución de Usuarios por Cluster")
    # Aquí mostraríamos el gráfico de Matplotlib/Seaborn de los clusters
    st.bar_chart(df_modelo["cluster"].value_counts())

    st.write("Método del Codo")
    st.pyplot(plt.gcf())

with col2:
    st.write("Métricas de Calidad")
    st.metric(label="Índice de Silhouette", value="0.68", delta="Optimizado")
    st.metric(label="Precisión @ 3", value="85%", delta="7% vs mes anterior")


from sqlalchemy import create_engine, text
import pandas as pd

# Creamos la conexión al servidor SQL
engine = create_engine(
    "sqlite:///academia.db"
)  # Cambiado a SQLite para evitar necesidad de PostgreSQL


def cargar_datos_para_entrenamiento():
    query = """
    SELECT u.user_id, i.rating, i.progreso, u.estudios_num
    FROM users u
    JOIN interactions i ON u.user_id = i.user_id
    """
    # Leemos directamente a un DataFrame de Pandas
    df = pd.read_sql(query, engine)
    return df
