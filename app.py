import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import numpy as np
from scipy.spatial import cKDTree

st.set_page_config(page_title="Coyhaique 15 min", layout="wide")
st.title(" Accesibilidad a 15 minutos — Coyhaique")
st.markdown(
    "¿Qué tan bien conectados están los barrios de Coyhaique a salud "
    "y bomberos en 1250 metros?"
)
@st.cache_data
def cargar_datos():
    # --- Límite urbano ---
    limite = gpd.read_file("data/Limites_Urbanos_PRC.geojson").to_crs(4326)

    # --- Infraestructura (hospitales + bomberos) ---
    infra = gpd.read_file("data/Hospitales.shp")

    # El shapefile viene en UTM 18S, lo pasamos a WGS84 para folium
    infra = infra.to_crs(4326)

    # La geometría viene como MultiPoint (un solo punto cada una) -> la explotamos a Point
    infra = infra.explode(index_parts=False).reset_index(drop=True)

    # Mapeo de categoría según el id
    mapa_categoria = {1: "Salud", 2: "Bomberos"}
    infra["categoria"] = infra["id"].map(mapa_categoria)

    # Etiqueta correlativa por categoría
    infra["etiqueta"] = (infra.groupby("categoria").cumcount() + 1).astype(str)
    infra["etiqueta"] = infra["categoria"] + " N°" + infra["etiqueta"]

    return limite, infra

limite, infra = cargar_datos()

# --- Verificación temporal en pantalla (la sacamos en el próximo bloque) ---
centro = limite.geometry.centroid.iloc[0]

# --- Sidebar: mapa base ---
st.sidebar.header("⚙️ Configuración del mapa")
mapa_base = st.sidebar.selectbox(
    "Mapa base", ["OpenStreetMap", "CartoDB Positron", "Satélite (Esri)"]
)
tiles_dict = {"OpenStreetMap": "OpenStreetMap", "CartoDB Positron": "CartoDB positron", "Satélite (Esri)": None}

# --- Sidebar: capas ---
st.sidebar.header("📍 Capas")
mostrar_limite = st.sidebar.checkbox("Límite urbano", value=True)
mostrar_salud = st.sidebar.checkbox("Salud", value=True)
mostrar_bomberos = st.sidebar.checkbox("Bomberos", value=True)

# --- Colores por categoría ---
colores = {"Salud": "#e41a1c", "Bomberos": "#377eb8"}

# --- Sidebar: búsqueda por atributo ---
st.sidebar.header("🔍 Búsqueda")
busqueda = st.sidebar.selectbox(
    "Buscar punto específico",
    ["Ninguno"] + sorted(infra["etiqueta"].tolist())
)

# --- Crear mapa base ---
if busqueda != "Ninguno":
    punto_buscado = infra[infra["etiqueta"] == busqueda].geometry.iloc[0]
    centro_mapa = [punto_buscado.y, punto_buscado.x]
    zoom_mapa = 17
else:
    centro_mapa = [centro.y, centro.x]
    zoom_mapa = 14

m = folium.Map(
    location=centro_mapa,
    zoom_start=zoom_mapa,
    tiles=tiles_dict[mapa_base] if tiles_dict[mapa_base] else None
)

if mapa_base == "Satélite (Esri)":
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satélite"
    ).add_to(m)

# --- Límite urbano ---
if mostrar_limite:
    folium.GeoJson(
        limite,
        name="Límite urbano",
        style_function=lambda x: {"fillColor": "none", "color": "black", "weight": 2}
    ).add_to(m)
    # --- Sidebar: superficie de accesibilidad ---
st.sidebar.header("🎯 Accesibilidad 15 min")
mostrar_accesibilidad = st.sidebar.checkbox("Mostrar superficie de accesibilidad", value=True)
incluir_salud = st.sidebar.checkbox("Incluir Salud en el cálculo", value=True)
incluir_bomberos = st.sidebar.checkbox("Incluir Bomberos en el cálculo", value=True)

# --- Sidebar: panel de estadísticas y filtro ---
st.sidebar.header("📊 Estadísticas")

capa_stats = st.sidebar.selectbox(
    "Ver estadísticas de:", ["Todas", "Salud", "Bomberos"]
)

if capa_stats == "Todas":
    subset_stats = infra
else:
    subset_stats = infra[infra["categoria"] == capa_stats]

st.sidebar.metric("Puntos totales", len(subset_stats))
st.sidebar.write(f"Salud: {(infra['categoria'] == 'Salud').sum()} | "
                  f"Bomberos: {(infra['categoria'] == 'Bomberos').sum()}")

umbral_filtro = st.sidebar.slider(
    "Filtrar: % mínimo de accesibilidad", 0.0, 1.0, 0.0, 0.05
)
if mostrar_accesibilidad and (incluir_salud or incluir_bomberos):

    # Trabajamos en metros (UTM 18S) para que las distancias sean reales, no en grados
    limite_utm = limite.to_crs(32718)
    infra_utm = infra.to_crs(32718)

    minx, miny, maxx, maxy = limite_utm.total_bounds
    resolucion = 30  # metros por celda de la grilla

    xs = np.arange(minx, maxx, resolucion)
    ys = np.arange(miny, maxy, resolucion)
    xx, yy = np.meshgrid(xs, ys)
    puntos_grilla = np.column_stack([xx.ravel(), yy.ravel()])

    DISTANCIA_15MIN = 1250  # metros (~15 min caminando a 5 km/h)

    def score_categoria(categoria_valor):
        coords = np.array([
            [geom.x, geom.y] for geom in infra_utm[infra_utm["categoria"] == categoria_valor].geometry
        ])
        arbol = cKDTree(coords)
        dist, _ = arbol.query(puntos_grilla)
        return np.clip(1 - dist / DISTANCIA_15MIN, 0, 1)

    scores = []
    if incluir_salud:
        scores.append(score_categoria("Salud"))
    if incluir_bomberos:
        scores.append(score_categoria("Bomberos"))

    score_final = np.mean(scores, axis=0).reshape(xx.shape)
    score_final = np.flipud(score_final)  # para que la imagen quede orientada correctamente

    # --- Convertir a imagen RGBA con rampa de color ---
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    cmap = cm.get_cmap("RdYlGn")
    rgba_img = cmap(score_final)
    rgba_img[..., 3] = np.where(score_final > 0, 0.6, 0)  # transparente donde no hay acceso

    # --- Límites geográficos de la imagen en WGS84 (folium necesita lat/lon) ---
    esquinas_utm = gpd.GeoSeries(
        [gpd.points_from_xy([minx, maxx], [miny, maxy])[0], gpd.points_from_xy([minx, maxx], [miny, maxy])[1]],
        crs=32718
    ).to_crs(4326)
    lon_min, lat_min = esquinas_utm.iloc[0].x, esquinas_utm.iloc[0].y
    lon_max, lat_max = esquinas_utm.iloc[1].x, esquinas_utm.iloc[1].y

    folium.raster_layers.ImageOverlay(
        image=rgba_img,
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        name="Accesibilidad 15 min",
        opacity=0.7,
    ).add_to(m)

    # --- Leyenda de la rampa de color ---
    leyenda_raster_html = """
    <div style="position: fixed; bottom: 30px; right: 30px; z-index:9999;
            background-color:#3b0764; color:white; padding:10px;
            border:2px solid #2e0550; border-radius:5px;">
<b>Accesibilidad (15 min)</b><br>
<div style="background: linear-gradient(to right, #a50026, #ffffbf, #006837); width:150px; height:15px;"></div>
<div style="display:flex; justify-content:space-between; width:150px; font-size:11px;">
    <span>0.0 (sin acceso)</span><span>1.0 (máx.)</span>
</div>
</div>
"""
    m.get_root().html.add_child(folium.Element(leyenda_raster_html))
    porcentaje_cobertura = (score_final >= umbral_filtro).sum() / score_final.size * 100
    st.sidebar.metric(
        f"Área con accesibilidad ≥ {umbral_filtro:.2f}",
        f"{porcentaje_cobertura:.1f}%"
    )
# --- Función para agregar puntos de una categoría ---
def agregar_puntos(gdf, categoria_valor, mostrar):
    if not mostrar:
        return
    subset = gdf[gdf["categoria"] == categoria_valor]
    for _, row in subset.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=8,
            color=colores[categoria_valor],
            fill=True,
            fill_color=colores[categoria_valor],
            fill_opacity=0.9,
            tooltip=f"<b>{row['etiqueta']}</b><br>Categoría: {row['categoria']}"
        ).add_to(m)

agregar_puntos(infra, "Salud", mostrar_salud)
agregar_puntos(infra, "Bomberos", mostrar_bomberos)

if busqueda != "Ninguno":
    fila = infra[infra["etiqueta"] == busqueda].iloc[0]
    folium.CircleMarker(
        location=[fila.geometry.y, fila.geometry.x],
        radius=14,
        color="yellow",
        fill=False,
        weight=4,
        tooltip=f"🔍 {fila['etiqueta']}"
    ).add_to(m)

folium.LayerControl().add_to(m)

# --- Leyenda fija ---
leyenda_html = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index:9999;
            background-color:black; padding:10px; border:2px solid grey; border-radius:5px;">
<b>Leyenda</b><br>
<span style="color:#e41a1c;">●</span> Salud<br>
<span style="color:#377eb8;">●</span> Bomberos
</div>
"""
m.get_root().html.add_child(folium.Element(leyenda_html))

# --- Mostrar mapa en la app ---
st_folium(m, width=1200, height=650)