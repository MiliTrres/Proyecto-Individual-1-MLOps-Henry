import pandas as pd
import requests
from bs4 import BeautifulSoup


def porcentaje_valores_nulos(dataframe):
    '''
    Función que recibe como argumento un DataFrame
    y retorna el porcentaje de valores nulos por columna.

    '''
    total_filas = dataframe.shape[0]
    porcentaje_nulos = (dataframe.isnull().sum() / total_filas) * 100
    
    for columna, porcentaje in porcentaje_nulos.items():
        print(f'La columna {columna} tiene un {porcentaje: .2f} % de valores nulos')


def tamanio_datsets(diccionario):
    '''
    Función que recibe como argumento un diccionario, y retorna
    el nombre del DataFrame y el tamaño del mismo.

    Recorremos un diccionario, con un ciclo for, cuya clave es el nombre del DataFrame y su valor, 
    es el DataFrame en si.

    '''
    for nombre, dataframe in diccionario.items():
        print('El tamaño del DataFrame', nombre, 'es:', dataframe.shape)

        
def deteccion_columnas_anidadas(df):
    '''
    Función que recibe como argumento un DataFrame, y retorna una lista
    con las colmunas anidadas del mismo.

    Creamos una lista vacia donde se iran almacenando los nombres de las columnas anidadas.
    Iteramos columna por columna con un ciclo for y convertimos el tipo de dato de la columna a string.
    Verificamos si al menos un valor de la columna comienza con '[{' o '{', y si lo hace, se agrega el nombre
    de la columna a la lista vacia creada inicialmente.
    
    '''

    columnas_anidadas = []

    for columna in df.columns:
        if any(df[columna].astype(str).str.startswith('[{')) or any(df[columna].astype(str).str.startswith('{')):
            columnas_anidadas.append(columna)
     
    return columnas_anidadas
    


def obtener_ano_lanzamiento(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            game_year_element = soup.find('div', {'class': 'date'})
            if game_year_element:
                game_year = game_year_element.text.strip()
                return game_year
            else:
                return None
        else:
            return None
    except Exception as e:
        print(f"Error en la URL {url}: {str(e)}")
        return None



