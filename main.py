 
from fastapi import FastAPI
import uvicorn
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi import HTTPException
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def incio ():
    principal= """
    <!DOCTYPE html>
    <html>
        <head>
            <title>API Steam</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    padding: 20px;
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                p {
                    color: #666;
                    text-align: center;
                    font-size: 18px;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <h1>API de consultas sobre juegos de la plataforma Steam</h1>
            <p>¡Bienvenido a la API de Steam!
            Puede hacer sus consultas en el siguiente link:</p>
            <a href="https://deploy-p1-milagros.onrender.com/docs">
            
            <p>Milagros Torres -</p>
        </body>
    </html>

        """    
    return principal



@app.get("/playtimegenre/{genero}")
async def PlayTimeGenre(genero: str):
    '''
    Función que recibe como parametro un genero (str), y retorna el año de lanzamiento con más horas jugadas para ese genero.
    Primero se filtran los años por genero, luego se agrupa por año y se suman las horas jugadas.
    Por ultimo, encuentra el año con más horas jugadas.
    items_games = pd.read_parquet('Data/df_funcion_1.parquet')

    genre_data = items_games[items_games['genres'].str.contains(genero, case=False, na=False)]

    # Agrupa por año y suma las horas jugadas
    genre_by_year = genre_data.groupby('release_year')['playtime_forever'].sum().reset_index()

    # Encuentra el año con más horas jugadas
    year_with_most_playtime = genre_by_year.loc[genre_by_year['playtime_forever'].idxmax()]

    return {"Año de lanzamiento con más horas jugadas para " + genero: year_with_most_playtime['release_year']}
    '''
    try:
        # Intenta cargar los datos
        items_games = pd.read_parquet('df_funcion_1.parquet')

        # Filtra los datos por género
        genre_data = items_games[items_games['genres'] == genero]

        # Agrupa por año y suma las horas jugadas
        genre_by_year = genre_data.groupby('release_year')['playtime_forever'].sum().reset_index()

        # Encuentra el año con más horas jugadas
        year_with_most_playtime = genre_by_year.loc[genre_by_year['playtime_forever'].idxmax()]

        result = {"Año de lanzamiento con más horas jugadas para " + genero: year_with_most_playtime['release_year']}
        
        return result
    except Exception as e:
        # Maneja la excepción y devuelve una respuesta de error
        error_message = "Ocurrió un error al procesar la solicitud: " + str(e)
        return JSONResponse(content={"error": error_message}, status_code=500)

@app.get("/userforgenre/{genero}")
async def UserForGenre(genero: str):

    items_games_2 = pd.read_parquet('Data/df_funcion_2.parquet')
    # Filtra los datos por género
    genre_data = items_games_2[items_games_2['genres'] == genero]

    # Agrupa por usuario y suma las horas jugadas
    user_playtime = genre_data.groupby('user_id')['playtime_forever'].sum().reset_index()

    # Encuentra el usuario con más horas jugadas
    user_with_most_playtime = user_playtime.loc[user_playtime['playtime_forever'].idxmax()]

    # Filtra los datos por usuario para calcular la acumulación de horas jugadas por año
    user_data = genre_data[genre_data['user_id'] == user_with_most_playtime['user_id']]
    playtime_by_year = user_data.groupby('release_year')['playtime_forever'].sum().reset_index()

    # Convierte los datos a un formato de lista de diccionarios y cambia los nombres de las claves
    playtime_by_year_list = playtime_by_year.rename(columns={'release_year': 'Año', 'playtime_forever': 'Horas'}).to_dict(orient='records')

    result = {
        "Usuario con más horas jugadas para " + genero: user_with_most_playtime['user_id'],
        "Horas jugadas": playtime_by_year_list
    }

    return result


@app.get("/usersrecommend/{anio}")
async def UsersRecommend(anio: int):

    reviews_games = pd.read_parquet('Data/df_funcion_3y4.parquet')

    reviews_year = reviews_games[reviews_games['posted'] == anio]

    # Filtra las reseñas recomendadas con sentimiento positivo o neutral
    recommended_games = reviews_year[(reviews_year['recommend'] == True) & (reviews_year['sentiment_analysis'].isin([1, 2]))]

    # Agrupa por juego y cuenta las recomendaciones
    top_games = recommended_games.groupby('app_name')['recommend'].count().reset_index()

    # Ordena los juegos en orden descendente de recomendaciones
    top_games = top_games.sort_values(by='recommend', ascending=False)

    # Toma los 3 juegos principales
    top_3_games = top_games.head(3)

    # Convierte los datos en el formato de retorno
    result = [{"Puesto " + str(i + 1): game} for i, game in enumerate(top_3_games['app_name'])]

    return result



@app.get("/usersnotrecommend/{anio}")
async def UsersNotRecommend(anio: int):

    reviews_games = pd.read_parquet('Data/df_funcion_3y4.parquet')

    reviews_year = reviews_games[reviews_games['posted'] == anio]

    # Filtra las reseñas recomendadas con sentimiento positivo o neutral
    not_recommended_games = reviews_year[(reviews_year['recommend'] == False) & (reviews_year['sentiment_analysis'] == 0)]

    # Agrupa por juego y cuenta las recomendaciones
    top_games = not_recommended_games.groupby('app_name')['recommend'].count().reset_index()

    # Ordena los juegos en orden descendente de recomendaciones
    top_games = top_games.sort_values(by='recommend', ascending=False)

    # Toma los 3 juegos principales
    top_3_games = top_games.head(3)

    # Convierte los datos en el formato de retorno
    result = [{"Puesto " + str(i + 1): game} for i, game in enumerate(top_3_games['app_name'])]

    return result

@app.get("/sentimentanalysis/{anio}")
async def SentimentAnalysis(anio: int):

    reviews_games_2 = pd.read_parquet('Data/df_funcion_5.parquet')

        # Filtra los datos por el año dado
    reviews_year = reviews_games_2[reviews_games_2['release_year'] == int(anio)]

    Negativos = 0
    Neutral = 0
    Positivos = 0

    for i in reviews_year["sentiment_analysis"]:
        if i == 0:
            Negativos += 1
        elif i == 1:
            Neutral += 1 
        elif i == 2:
            Positivos += 1

    result = {"Negative": Negativos , "Neutral" : Neutral, "Positive": Positivos}
    return result



@app.get("/gamerecommendation/{id}")
async def GameRecommendation(id: int):
    cosine_sim = np.load('Data/similitud.npy')

    steam_games_final = pd.read_parquet('Data/steamgames_items_items.parquet')

    idx = steam_games_final[steam_games_final['id'] == id].index[0]

    rec_indices = cosine_sim[idx]
    rec_games = steam_games_final.iloc[rec_indices]['app_name']


    recomendaciones = []  # Lista para almacenar las recomendaciones

    for count, game_id in enumerate(rec_games, start=1):
        recomendaciones.append(f'Número {count}: {game_id}')

        # Limitar a 5 recomendaciones
        if count == 5:
            break

    return {'TOP 5 juegos similares:': recomendaciones }
