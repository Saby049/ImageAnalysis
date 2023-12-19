import os
import requests
import base64
from flask_cors import CORS, cross_origin
from flask import Flask, request
from openai import AzureOpenAI
import json

# Configuration
GPT4V_KEY = "bfdc6726bd7f4dbd8bbe701c8c55a138"

# setting app and CORS
app = Flask(__name__)
CORS(app)


@app.route('/image_observation', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def image_observation():
    text = request.data.decode('utf-8')
    message_text = [{"role":"system","content":"You are an AI assistant which takes multiple sentences and club them into one sentence and make it better for the audience. If in the input it is mentioned that it is a blurry image mention it , otherwise do not mention about the quality of the image. Installation Rating is best 5 and worst 1 , it will be in the range of 5 to 1."}]
    message_text.append({"role":"user","content":text})
    # gets the API Key from environment variable AZURE_OPENAI_API_KEY
    client = AzureOpenAI(
        api_version= "2023-07-01-preview",
        azure_endpoint= "https://cloudxp-openai.openai.azure.com",
        api_key= "53fb99eee193407081d9d28d1ab7eabd",
    )

    completion = client.chat.completions.create(
        model="gpt-35-turbo",  # e.g. gpt-35-instant
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
                        "text": "You are an AI assistant that explains devices and suggests whether proper installation has been provided or not. If not suggest right options. Dont provide any more details. Provide only image description and improvements."
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
                        "text": "Provide Image description and suggetion"
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800
    }

    GPT4V_ENDPOINT = "https://visionopenai007.openai.azure.com/openai/deployments/visiongpt/extensions/chat/completions?api-version=2023-07-01-preview"
    # Send request
    try:
        response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
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