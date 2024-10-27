import os
from groq import Groq

#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

client = Groq(
    api_key='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',
)

name = input("Enter the name of the historical figure you want to talk to: ")

completion = client.chat.completions.create(
    model="llama-3.2-3b-preview",
    messages=[
        {
            "role": name,
            "content": ""
        }
    ],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
