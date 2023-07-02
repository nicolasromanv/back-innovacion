import requests
import json

def obtener_caratula(juego):
    url = "https://api.igdb.com/v4/games"
    parametros = {
        "search": juego,
        "fields": "cover.url",
        "limit": 1
    }
    headers = {
        "Client-ID": "rkml50dibr3y5xa3hidmk4yfh5qxxx",  # Reemplaza esto con tu propio Client-ID de IGDB
        "Authorization": "Bearer qui17ip45k52pmprlr4zqmp9atr81t"  # Reemplaza esto con tu propio Access Token de IGDB
    }

    response = requests.get(url, params=parametros, headers=headers)
    data = response.json()

    if data:
        cover_url = data[0]["cover"]["url"]
        return {"caratula": cover_url}
    else:
        return {"error": "No se encontró la carátula del juego."}

#nombre_juego = input("Ingresa el nombre del juego: ")

