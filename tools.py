from pathlib import Path
import os, json
from langchain.tools import tool
from dotenv import load_dotenv
import requests

load_dotenv()

crypto_api_key = os.getenv("CRYPTO_API_KEY")
API_URL_SYMBOL = 'https://api.freecryptoapi.com/v1/getData?symbol='
API_KEY = crypto_api_key

path = Path(__file__).parent / "kb.json"


@tool
def kb(query: str) -> str:
    """
    Fetch crypto coin data from the local knowledge base.
    
    Args:
        query: The name or symbol of the cryptocurrency to search for (e.g., 'bitcoin', 'BTC', 'ethereum')
    
    Returns:
        JSON string with coin data or error message
    """
    try:
        with open(path, "r") as f:
            kb_data = json.load(f)
        
        query_lower = query.lower().strip()
        
        for coin in kb_data["coins"]:
            coin_name = coin.get("coin", "").lower()
            coin_symbol = coin.get("symbol", "").lower()
            
            if coin_name == query_lower or coin_symbol == query_lower:
                return json.dumps(coin)
        
        return json.dumps({"error": "NOT_FOUND_IN_KB"})
    
    except Exception as e:
        return json.dumps({"error": f"KB lookup failed: {str(e)}"})


@tool
def get_crypto_price(query: str) -> str:
    """
    Fetch real-time crypto coin data from external API when data is not available in KB.
    
    Args:
        query: The name or symbol of the cryptocurrency to search for (e.g., 'bitcoin', 'BTC', 'ethereum')
    
    Returns:
        JSON string with current price data or error message
    """
    try:
        coin_name = query.lower().strip()
        
        try:
            symbol_api = 'https://api.coingecko.com/api/v3/coins/' + coin_name  
            response = requests.get(symbol_api, timeout=5)
            symbol_data = response.json()
            coin_symbol = symbol_data['symbol'].upper()
            API_URL = API_URL_SYMBOL + f'{coin_symbol}'
        except (KeyError, requests.exceptions.RequestException):
            API_URL = API_URL_SYMBOL + f'{coin_name.upper()}'

        response = requests.get(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}"
            },
            params={
                "coin": coin_name,
                "currency": "usd"
            },
            timeout=10
        )

        response.raise_for_status()
        
        api_data = response.json()
        
        if 'symbols' in api_data and len(api_data['symbols']) > 0:
            final_response = api_data['symbols'][0]
            
            try:
                with open(path, 'r') as f:
                    kb_data = json.load(f)
                
                kb_data['coins'].append(final_response)
                
                with open(path, 'w') as f:
                    json.dump(kb_data, f, indent=2)
            except Exception as kb_error:
                print(f"Warning: Could not update KB: {kb_error}")
            
            return json.dumps(final_response)
        else:
            return json.dumps({"error": "Coin data not found in API response"})
            
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"API request failed: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch crypto price: {str(e)}"})