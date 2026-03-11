import os
from ollama import Client

OLLAMA_API_KEY="2d13d8be27e742499f4ffc53edde3554.vn48Y4wV8VzacTux5rh03qAU"

client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + OLLAMA_API_KEY}
)

prompt = "Test per vedere se funziona"

response = client.chat(
    model='qwen3.5:latest',
    messages=[{"role": "user", "content": prompt}],
    stream=False,
    options={"temperature": 0.3},
)

print(response)