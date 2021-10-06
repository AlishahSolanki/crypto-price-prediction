#API


import requests

headers ={
    'X-CMC_PRO_API_KEY' : '051215f6-3991-458d-8d69-0827c233befe',
    'Accepts' : 'Application/json'
    }

params = {
    'start':'1',
    'limit':'5',
    'convert':'USD'
   
    }

url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'

json = requests.get(url, params=params, headers=headers).json()

print(json)

