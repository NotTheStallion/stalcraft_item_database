import os
import requests
from dotenv import load_dotenv
import time
import functools

load_dotenv()



def timer(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = time.perf_counter() - start
        print(f"{func.__name__} completed for {runtime:.4f} secs")
        return result
    return _wrapper

def sleep(timeout, retry=3):
    def the_real_decorator(function):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < retry:
                newtimeout = timeout + retries * 5
                try: 
                    value = function(*args, **kwargs)
                except:  # noqa: E722
                    retries += 1
                    if retries != retry:
                        print(f'Sleeping for {newtimeout} seconds | Remaining retries {retry - retries}')
                        time.sleep(newtimeout) 
                        print('Retrying...') 
                else:
                    return value
        return wrapper
    return the_real_decorator

API_TOKEN_URL = "https://exbo.net/oauth/token"
HISTORY_URL = "https://eapi.stalcraft.net/RU/auction/{item}/history"
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

def get_token():
    resp = requests.post(API_TOKEN_URL, json={
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
        "scope": ""
    })
    return resp.json()["access_token"]

ACCESS_TOKEN = get_token()
HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

def retry_on_429(timeout=5, max_retries=10):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while True:
                response = func(*args, **kwargs)
                # If response is a requests.Response object, check status code
                if hasattr(response, 'status_code') and response.status_code == 429:
                    retries += 1
                    if retries > max_retries:
                        print(f"Max retries ({max_retries}) reached for 429 responses.")
                        return response
                    print(f"Received 429, sleeping for {timeout} seconds (retry {retries}/{max_retries})...")
                    time.sleep(timeout)
                else:
                    return response
        return wrapper
    return decorator

@retry_on_429(timeout=5, max_retries=10)
def _request_with_token(item_id):
    global ACCESS_TOKEN, HEADERS
    url = HISTORY_URL.format(item=item_id)
    r = requests.get(url, headers=HEADERS, params={"limit": 1, "additional": "true"})
    if r.status_code == 401:
        ACCESS_TOKEN = get_token()
        HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
        r = requests.get(url, headers=HEADERS, params={"limit": 1, "additional": "true"})
    return r

def is_id_valid(item_id):
    r = _request_with_token(item_id)
    if r.ok :
        return r.json()
    return None


if __name__ == "__main__":
    # Example usage:
    print(is_id_valid("4qdqn"))
    print(is_id_valid("abcde"))
    for _ in range(5000):
        print(is_id_valid("4qdqn"))
        print(is_id_valid("abcde"))
        
        