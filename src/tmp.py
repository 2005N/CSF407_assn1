import os
from mistralai import Mistral

# Set your API key as an environment variable for security
os.environ["MISTRAL_API_KEY"] = "T8VEqzz4oYeK3affFYmu09iiBpAZJgvQ"

# Retrieve the API key from the environment variable
api_key = os.environ["MISTRAL_API_KEY"]

# Specify the model you want to use
model = "mistral-large-latest"

# Initialize the Mistral client with your API key
client = Mistral(api_key=api_key)

# Send a chat request to the API
chat_response = client.chat.complete(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "What is the best French cheese?",
        },
    ]
)

# Print the response from the model
print(chat_response.choices[0].message.content)
