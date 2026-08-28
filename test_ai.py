import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "Say hello and confirm that you are connected to my Multi-Domain AI Analytics project."
        }
    ]
)

print(response["message"]["content"])