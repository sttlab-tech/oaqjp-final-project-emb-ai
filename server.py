"""
Flask routes for the Emotion detection web app
"""

from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask("Emotion Detector")

@app.route("/emotionDetector")
def detector():
    """
    This function manages the analysis of emotions in a text string
    """

    # Retrieve the text to analyze from the request arguments
    text_to_analyze = request.args.get('textToAnalyze')

    # Pass the text to the emotion_detector function and store the response
    response = emotion_detector(text_to_analyze)

    # Extract the label and score from the response
    anger = response['anger']
    disgust = response['disgust']
    fear = response['fear']
    joy = response['joy']
    sadness = response['sadness']
    dominant_emotion = response['dominant_emotion']

    if dominant_emotion is None:
        # Return an error message
        return "Invalid text! Please try again!"
    else:
        # Return a formatted string with the sentiment label and score
        return f"""For the given statement, the system response is 
            'anger': {anger}, 
            'disgust': {disgust}, 
            'fear': {fear}, 
            'joy': {joy} and 
            'sadness': {sadness}. 
            The dominant emotion is {dominant_emotion}."""

@app.route("/")
def render_index_page():
    """
    This function routes trafic to the main web app page
    """

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

