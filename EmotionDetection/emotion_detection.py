"""
Emotion detection module, which provides a set of functions to detect emotions in text content.
"""

import requests, json

def emotion_detector(text_to_analyse): 
    """
        Function that takes a text string and performs a emotion analysis using the Watson NLP API.
        Returns a dict that contains a label and a score between -1 and +1.
    """

    # URL of the sentiment analysis service
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'  

    # Payload that is sent to the Watson NLP API, with the text_to_analyse string variable received in input of the function
    payload = { "raw_document": { "text": text_to_analyse } }  

    # HTTP headers that are sent to the Watson NLP API
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

    # Call to the Watson NLP API to perform the emotion analysis on text_to_analyse
    response = requests.post(url, json = payload, headers=headers)  


    # Parsing of the API response
    formatted_response = json.loads(response.text)

    # Management of invalid API request
    if response.status_code == 400:
        # Formatting of the JSON response
        emotions = {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }
        return emotions

    # Formatting of the JSON response
    emotions = {
        "anger": formatted_response['emotionPredictions'][0]['emotion']['anger'],
        "disgust": formatted_response['emotionPredictions'][0]['emotion']['disgust'],
        "fear": formatted_response['emotionPredictions'][0]['emotion']['fear'],
        "joy": formatted_response['emotionPredictions'][0]['emotion']['joy'],
        "sadness": formatted_response['emotionPredictions'][0]['emotion']['sadness']
    }

    dominant_emotion = max(emotions, key=emotions.get)
    emotions["dominant_emotion"] = dominant_emotion



    # Return the response text from the API
    return emotions  
