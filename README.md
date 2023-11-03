# Proyecto individual n° 1:
# Recomendación de Juegos Steam con MLOps


Este proyecto representa el Producto Individual 1 en el campo de Machine Learning, desarrollado dentro del bootcamp de Henry. Su propósito es simular el rol de un MLOps Engineer, una figura que combina las habilidades de un Data Engineer y un Data Scientist. En esta simulación, el MLOps Engineer trabaja para la conocida plataforma de videojuegos, Steam. El proyecto se centra en abordar un desafío empresarial clave: la creación de un Producto Mínimo Viable (MVP) que incorpora tanto una API de implementación, como un modelo de Machine Learning.

## Descripción del proyecto

El proyecto busca solucionar dos problemas esenciales en la plataforma Steam:

**Análisis de Sentimientos de Usuarios:** El primer desafío consiste en analizar y clasificar los comentarios de los usuarios. Para ello, se emplea la librería TextBlob, una herramienta de Procesamiento del Lenguaje Natural (NLP). TextBlob determina la polaridad del sentimiento en cada comentario y lo clasifica como negativo, neutral o positivo.

**Sistema de Recomendación de Juegos:** El segundo desafío, radica en construir un sistema de recomendación de videojuegos. Este sistema proporciona recomendaciones de juegos a los usuarios basándose en sus preferencias y comportamientos anteriores.

## Datos

Para desarrollar el proyecto se utilizaron tres archivos en formato JSON:

**output_steam_games.json**: Contiene información sobre los juegos, como el nombre, el editor, el desarrollador, los precios y las etiquetas.

**australian_users_items.json**: Contiene información sobre los juegos utilizados por los usuarios y el tiempo que cada usuario pasa en cada juego.

**australian_users_reviews.json**: Contiene los comentarios que los usuarios realizaron sobre los juegos que usan, así como recomendaciones o críticas sobre esos juegos, ID del usuario y su URL.

## Tareas realizadas

### ETL (Extract, Transform and Load)

En esta fase, se realizo el Notebook ETL. Se extrajeron datos de los DataFrames iniciales para familiarizarse con ellos y comenzar la limpieza de datos. Los datos limpios se almacenaron en formato CSV.

### Feature Engineering

La ingeniería de características se centró en el análisis de sentimientos de los comentarios de los usuarios, usando la librería TextBlob. Además, se prepararon los conjuntos de datos necesarios para optimizar las consultas y funcionalidades del servicio en la nube.

### EDA (Exploratory Data Analysis)

Se llevó a cabo un análisis exploratorio de los tres conjuntos de datos después del proceso de ETL. Esto permitió visualizar mejor las variables categóricas y numéricas, identificando las que son esenciales para el modelo de recomendación.

### Desarrollo de la API

Se construyó una API mediante el uso del framework FastAPI. Esta API ofrece varias funciones, estas son: 

- **PlayTimeGenre:** Esta función recibe como parametro un genero de juego y retorna el año con más horas jugadas para ese genero.

- **UserForGenre:** Esta función recibe como parametro un genero de juego y retorna el usuario con más horas jugadas para dicho genero, y una lista de aculuación de horas jugadas para dicho genero.

- **UsersRecommend:** Esta función recibe como parametro un año y retorna el top 3 de los juegos más recomendados para dicho año.

- **UsersNotRecommend:** Esta función recibe como parametro un año y retorna el top 3 de los juegos menos recomendados para dicho año.

- **SentimentAnalysis:** Esta función recibe como parametro un año y retorna una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.

- **GameRecommendation:** Esta función recibe como parametro el id de un juego y retorna una lista de los 5 juegos recomendados similares al ingreso.

### Modelamiento (Desarrollo de modelos de aprendizaje automático)

Para el desarrollo del Sistema de Recomendación, usamos el dataset resultante de etapas anteriores y creamos otro que contiene:

- **id**: ID del juego.
- **app_name**: Nombre del juego.
- **tags**: Etiquetas del juego.


### Despliegue

La API se desplegó en Render.com y está disponible en []().

## Video

Para obtener una explicación y demostración del funcionamiento de la API, consulta el [vídeo]().
