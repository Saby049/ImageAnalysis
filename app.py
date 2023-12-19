import os
import requests
import base64
from flask_cors import CORS, cross_origin
from flask import Flask, request
from openai import AzureOpenAI
import json

# Configuration
GPT4V_KEY = "your GPT4 key"

# setting app and CORS
app = Flask(__name__)
CORS(app)


@app.route('/image_observation', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def image_observation():
    text = request.data.decode('utf-8')
    message_text = [{"role":"system","content":"You are an AI assistant who takes multiple sentences and clubs them into one sentence and makes it better for the audience. If in the input it is mentioned that it is a blurry image mention it, otherwise do not mention the quality of the image. Installation Rating is best 5 and worst 1, it will be in the range of 5 to 1."}]
    message_text.append({"role":"user","content":text})
    # gets the API Key from environment variable AZURE_OPENAI_API_KEY
    client = AzureOpenAI(
        api_version= "your_version",
        azure_endpoint= "your_endpoint",
        api_key= "your_key",
    )

    completion = client.chat.completions.create(
        model="your_model",  # e.g. gpt-35-instant
        messages=message_text
    )
    result = completion.json()
    result_dict = json.loads(result)
    print(result_dict['choices'][0]['message']['content'])
    return "Image Observation: "+ result_dict['choices'][0]['message']['content']


# Process the info related input from the user
@app.route('/analyze_image', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def process_info():
    image = request.files['image']
    image.save(os.path.join('./images/', image.filename))
    encoded_image = base64.b64encode(open(os.path.join('./images/', image.filename), 'rb').read()).decode('ascii')
    headers = {
    "Content-Type": "application/json",
    "api-key": GPT4V_KEY,
    }
    payload = {
        "enhancements": {
            "ocr": {
                "enabled": True
            },
            "grounding": {
                "enabled": True
            }
        },
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI assistant that explains devices and suggests whether proper installation has been provided or not. If not suggest the right options. Don't provide any more details. Provide only image description and improvements."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Provide Image description and suggestion"
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800
    }

    GPT4V_ENDPOINT = "your_gpt4_endpoint"
    # Send request
    try:
        response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returns an unsuccessful status code
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    # Handle the response as needed (e.g., print or process)
    # Parse the JSON response
    response_json = response.json()

    # Extract the assistant content
    assistant_content = response_json['choices'][0]['message']['content']
    return assistant_content

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
