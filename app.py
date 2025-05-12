
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Configurar la página
st.set_page_config(
    page_title="Streamlit",
    page_icon="🏠",
    layout="wide"
)

# Funciones para cargar datos y modelos
@st.cache_data
def load_data(data):
    try:
        return pd.read_csv(data)
    except FileNotFoundError:
        st.error("No se encontró el archivo de datos.")
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/housing_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo. Asegúrate de que los archivos existen en la carpeta models/.")
        return None, None


# Título de la aplicación
st.title("Aplicativo funcional utilizando Streamlit")
st.markdown("Esta aplicación está diseñada para realizar análisis exploratorio y agrupamiento de datos tabulares cargados por el usuario.")

# Sidebar para navegación
page = st.sidebar.radio("Navegación", ["Inicio", "Limpieza de los datos", "Análisis de los datos", "Modelado sobre los datos"])

# Página de inicio
if page == "Inicio":
    st.header("Limpieza de datos")
    st.markdown("Sube tu archivo CSV para comenzar el análisis de datos.")
    uploaded_file = st.file_uploader("Carga un archivo CSV", type=["csv"])
    if uploaded_file is not None:     
        st.session_state.df = load_data(uploaded_file)
        st.session_state.original_df = st.session_state.df.copy()
        df = st.session_state.df
        if df is not None:
            uploaded_file = None
        else:
            st.error("No se pudo cargar el archivo. Por favor, verifica el formato y el contenido.")
    
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        st.success("Archivo cargado exitosamente.")
        st.subheader("Vista previa de los datos")
        st.dataframe(df.head())

# Página de "Análisis de los datos"
elif page == "Análisis de los datos":
    st.header("Análisis de los datos")
 
    # Verificar si los datos están cargados
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        numeric_df = df.select_dtypes(include=['number'])

        st.markdown("""
        Esta sección muestra diferentes visualizaciones de los datos para entender mejor las relaciones
        entre las variables y su impacto.
        """)

        # Resumen estadístico
        st.subheader("Resumen Estadístico")
        st.dataframe(numeric_df.describe())

        # Valores nulos
        st.subheader("Valores Nulos")
        st.dataframe(numeric_df.isnull().sum().to_frame("Valores Nulos"))

        # Distribución de datos
        st.subheader("Distribución de Datos")
        selected_column = st.selectbox("Selecciona una columna para el histograma", options=numeric_df.columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(numeric_df[selected_column], kde=True, ax=ax)
        ax.set_title(f"Distribución de {selected_column}")
        st.pyplot(fig)
        
        # Matriz de correlación
        st.subheader("Matriz de Correlación")
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, annot=True, fmt='.2f')
        st.pyplot(fig)

        # Análisis de Componentes Principales (PCA)
        st.subheader("Análisis de Componentes Principales (PCA)")
        n_components = st.slider("Número de componentes", 2, min(len(numeric_df.columns), 5), 2)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(numeric_df.dropna())
        pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])
        st.dataframe(pca_df.head())
        
        # Diagrama de Dispersión
        st.subheader("Diagrama de Dispersión") 
        col1, col2 = st.columns(2)      
        with col1:
            x_var = st.selectbox("Variable X", options=df.columns[1:].tolist())     
        with col2:
            y_var = st.selectbox("Variable Y", options=df.columns[1:].tolist(), index = 2)          
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=x_var, y=y_var, data=df, ax=ax)
        ax.set_title(f'Relación entre {x_var} y {y_var}')
        st.pyplot(fig)

        # Análisis de Clusters
        st.subheader("Análisis de Clusters (K-Means)")
        n_clusters = st.slider("Número de clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(numeric_df.dropna())
        numeric_df["Cluster"] = clusters
        st.dataframe(numeric_df.groupby("Cluster").mean())

    else:
        st.error("No se pueden mostrar visualizaciones sin datos.")

    
# Página de Limpieza
elif page == "Limpieza de los datos":
    st.header("Limpieza de los datos")

    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("Cambio de Tipo de Datos")
        col_to_convert = st.selectbox("Selecciona una columna para cambiar el tipo de dato", df.columns)
        current_dtype = df[col_to_convert].dtype
        st.markdown(f"**Tipo de dato actual:** {current_dtype}")
        new_dtype = st.selectbox("Selecciona el nuevo tipo de dato", ["Categorico", "Numerico"])
        if st.button("Convertir Tipo"):
            if new_dtype == "Categorico":
                df[col_to_convert] = df[col_to_convert].astype(str)
            elif new_dtype == "Numerico":
                df[col_to_convert] = pd.to_numeric(df[col_to_convert], errors='coerce')
            st.session_state.df = df
            st.success(f"La columna '{col_to_convert}' se ha convertido a {new_dtype} correctamente.")

    
        with st.form("data_cleaning_form"):
            remove_na = st.checkbox("Eliminar Filas con Valores Nulos")
            remove_duplicates = st.checkbox("Eliminar Filas Duplicadas")
            submit_button = st.form_submit_button("Aplicar Cambios")

        if submit_button:
            if remove_na:
                df.dropna(inplace=True)
            if remove_duplicates:
                df.drop_duplicates(inplace=True)
            st.session_state.df = df
            st.success("Cambios aplicados correctamente.")

        # Botón para deshacer cambios
        if st.button("Deshacer todos los cambios"):
            st.session_state.df = st.session_state.original_df.copy()
            st.success("Todos los cambios han sido deshechos correctamente.")

        # Mostrar datos limpios
        st.subheader("Vista previa de los datos")
        st.dataframe(df.head())
        # Mostrar tipos de datos
        st.subheader("Tipos de datos")
        st.dataframe(df.dtypes.to_frame(name="Tipo de Dato"))

    else:
        st.error("No se pueden limpiar los datos sin cargarlos. Ve a la pestaña de Inicio para cargar un archivo CSV.")

# Página de predicción
elif page == "Modelado sobre los datos":
    st.header("Modelado sobre los datos")

    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        numeric_df = df.select_dtypes(include=['number']).dropna()

        # Información sobre el tipo de modelado
        st.info(
            "**⚠️ Nota Importante:** Este módulo utiliza únicamente **Regresión Lineal** para el modelado de datos. "
            "Este método es adecuado cuando se espera una relación lineal entre las variables predictoras (features) y la variable objetivo. "
            "Para obtener los mejores resultados, los datos deben cumplir con ciertos supuestos, incluyendo:\n"
            "- Relación lineal entre las variables independientes y la variable objetivo.\n"
            "- Homocedasticidad (varianza constante del error).\n"
            "- Ausencia de multicolinealidad entre las variables independientes.\n"
            "- Normalidad de los errores."
        )

        # Selección de variable objetivo y features
        st.subheader("Selección de Variables")
        all_columns = numeric_df.columns.tolist()

        target_var = st.selectbox("Selecciona la variable objetivo (y)", all_columns)
        feature_columns = st.multiselect("Selecciona las columnas de features (X)", all_columns, default=[col for col in all_columns if col != target_var])

        # Dividir datos en entrenamiento y prueba
        st.subheader("División de los Datos")
        test_size = st.slider("Tamaño del conjunto de prueba (%)", 10, 50, 20)
        random_state = st.slider("Semilla aleatoria", 1, 100, 42)

        X = numeric_df[feature_columns]
        y = numeric_df[target_var]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)
        st.write("Tamaño del conjunto de entrenamiento:", X_train.shape)
        st.write("Tamaño del conjunto de prueba:", X_test.shape)

        # Escalado de las variables
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Entrenamiento del modelo de regresión lineal
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Predicciones
        y_pred = model.predict(X_test_scaled)

        # Métricas de evaluación
        st.subheader("Métricas de Evaluación")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Error Cuadrático Medio (MSE): {mse:.2f}")
        st.write(f"Coeficiente de Determinación (R²): {r2:.2f}")

        # Gráfico de predicciones vs valores reales
        st.subheader("Gráfico de Predicciones vs Valores Reales")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_title("Predicciones vs Valores Reales")
        ax.set_xlabel("Valores Reales")
        ax.set_ylabel("Predicciones")
        st.pyplot(fig)
    else:
        st.error("No se pueden entrenar modelos sin datos. Ve a la pestaña de Inicio para cargar un archivo CSV.")
