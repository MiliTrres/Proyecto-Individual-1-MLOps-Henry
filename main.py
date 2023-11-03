from fastapi import FastAPI
import uvicorn
import pandas as pd
#from fastapi.responses import HTMLResponse
#from fastapi import HTTPException
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

items_games = pd.read_parquet('Data/df_funcion_1.parquet')
items_games_2 = pd.read_parquet('Data/df_funcion_2.parquet')
reviews_games_2 = pd.read_parquet('Data/df_funcion_5.parquet')
steam_games_final = pd.read_parquet('Data/steamgames_items_items.parquet')

@app.get("/playtimegenre/{genero}")
async def PlayTimeGenre(genero: str):
    '''
    Función que recibe como parametro un genero (str), y retorna el año de lanzamiento con más horas jugadas para ese genero.
    Primero se filtran los años por genero, luego se agrupa por año y se suman las horas jugadas.
    Por ultimo, encuentra el año con más horas jugadas.
    
    '''
    genre_data = items_games[items_games['genres'].str.contains(genero, case=False, na=False)]

    # Agrupa por año y suma las horas jugadas
    genre_by_year = genre_data.groupby('release_year')['playtime_forever'].sum().reset_index()

    # Encuentra el año con más horas jugadas
    year_with_most_playtime = genre_by_year.loc[genre_by_year['playtime_forever'].idxmax()]

    return {"Año de lanzamiento con más horas jugadas para " + genero: int(year_with_most_playtime['release_year'])}


@app.get("/userforgenre/{genero}")
async def UserForGenre(genero: str):
    # Filtra los datos por género
    genre_data = items_games_2[items_games_2['genres'].str.contains(genero, case=False, na=False)]

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

    idx = steam_games_final[steam_games_final['id'] == float(id)].index[0]

    rec_indices = cosine_sim[idx]
    rec_games = steam_games_final.iloc[rec_indices]['app_name']

    print(f'TOP 5 juegos similares a {id}:')
    print('-----' * 8)

    recomendaciones = []  # Lista para almacenar las recomendaciones

    for count, game_id in enumerate(rec_games, start=1):
        recomendaciones.append(f'Número {count}: {game_id}')

        # Limitar a 5 recomendaciones
        if count == 5:
            break

    return recomendaciones 
