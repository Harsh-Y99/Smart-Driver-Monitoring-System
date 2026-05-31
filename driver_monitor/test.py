import requests

TOKEN = "8524244191:AAGvGJkun_14ey12NOZFNZzA_anyo10uVHs"
CHAT_ID = "1881450187"

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

res = requests.post(url, data={
    "chat_id": CHAT_ID,
    "text": "🔥 Driver system connected successfully!"
})

print(res.text)