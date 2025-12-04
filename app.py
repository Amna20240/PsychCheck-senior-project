from flask import Flask, request, jsonify
from flask_cors import CORS
import os, re
import joblib
import numpy as np
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# ================================
#  LOAD TRAINED ML MODELS
# ================================


anxiety_model = joblib.load("FINAL_anxiety_model.pkl")
depression_model = joblib.load("FINAL_depression_model.pkl")
dep_encoder = joblib.load("FINAL_dep_encoder.pkl")

# ================================
#  OPENAI CLIENT
# ================================
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ================================
#  KEYWORD PATTERNS
# ================================
RELATED_PATTERNS = [
    r"\banx(ious|iety)?\b", r"\bstress(ed)?\b", r"\bpanic\b",
    r"\bover[- ]?think(ing)?\b", r"\bworr(y|ied|ying)\b",
    r"\bdepress(ed|ion)?\b", r"\bsad|hopeless|worthless|empty|numb\b",
    r"\blonely|alone\b", r"\bsleep|insomnia|nightmare(s)?\b",
    r"\bfocus|concentrat(e|ion)\b", r"\bcope|coping|deal with\b",
    r"\bbreath(ing)?\b", r"\bground(ing)?\b", r"\bmood|motivat(e|ion)\b",
    r"\bcalm|relax(ation)?\b", r"\bpsych(check)?|psychology|therap(y|ist)\b",
    r"\bcounsel(l?or|ling)\b", r"\bburn ?out\b",
    r"\btired|exhausted|drained\b", r"\bscared|afraid|fear|nervous\b",
    r"\bself[- ]?esteem\b", r"\bmindful(ness)?\b",
    r"\bfeel\b", r"\bcan'?t sleep\b", r"\bfeel (down|low|off)\b"
]

CRISIS_PATTERNS = [
    "suicid", "kill myself", "hurt myself", "self-harm", "self harm"
]


def is_related(msg):
    return any(re.search(pat, msg, re.IGNORECASE) for pat in RELATED_PATTERNS)


def mentions_crisis(msg):
    return any(re.search(pat, msg, re.IGNORECASE) for pat in CRISIS_PATTERNS)


# ================================
# SYSTEM MESSAGE TO OPENAI
# ================================
SYSTEM_MESSAGE = (
    "You are PsychCheck, a calm and supportive mental wellbeing assistant. "
    "You help with anxiety, stress, depression, mood, coping skills, and emotional support. "
    "If the user mentions self-harm, urgently direct them to emergency help. "
    "If the question is unrelated, politely say you only help with emotional wellbeing. "
    "Your tone must be warm, simple, and caring."
)


# ================================
# CHAT ENDPOINT
# ================================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"response": "Can you tell me what’s on your mind?"})

    msg = user_message.lower()

    # Greetings
    if msg in ["hi", "hello", "hey"]:
        return jsonify({"response": "Hello , How are you feeling today?"})

    if "how are you" in msg:
        return jsonify({"response": "I'm here for you. How have you been feeling lately?"})

    if msg in ["yes", "yeah", "ok", "okay", "sure"]:
        return jsonify({"response": "Alright, go ahead. I'm listening."})

    if msg in ["no", "not really"]:
        return jsonify({"response": "That's okay. Share whenever you feel ready."})

    if "thank" in msg:
        return jsonify({"response": "You're always welcome. How are you feeling now?"})

    if msg in ["bye", "goodbye"]:
        return jsonify({"response": "Take care , I'm here anytime you need to talk."})

    # Crisis check
    if mentions_crisis(msg):
        return jsonify({
            "response":
                "I’m really sorry you feel like this. Please reach out to emergency services "
                "or someone you deeply trust **right now**. You deserve support immediately."
        })

    # Irrelevant topic
    if not is_related(msg):
        return jsonify({
            "response": "I can only help with emotional wellbeing, stress, anxiety, depression, and coping support."
        })

    # Send to OpenAI
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=300
        )
        reply = completion.choices[0].message.content.strip()
        return jsonify({"response": reply})

    except Exception as e:
        print("OpenAI error:", e)
        return jsonify({"response": "Sorry, I couldn’t process that. Try again in a moment."}), 500


# ================================
# ANXIETY ML PREDICTION
# ================================
@app.route("/predict/anxiety", methods=["POST"])
def predict_anxiety():
    data = request.json or {}

    features = np.array([[float(data.get(col, 0)) for col in [
        "Social interaction",
        "Mood level",
        "Stress level",
        "Negative thoughts",
        "Concentration problems",
        "Feeling lonely"
    ]]])

    prediction = anxiety_model.predict(features)[0]
    result = "Anxiety" if int(prediction) == 1 else "No Anxiety"

    return jsonify({"prediction": result})


# ================================
# DEPRESSION ML PREDICTION
# ================================
@app.route("/predict/depression", methods=["POST"])
def predict_depression():
    data = request.json or {}

    features = np.array([[float(data.get(col, 0)) for col in [
        "Social interaction",
        "Mood level",
        "Stress level",
        "Negative thoughts",
        "Concentration problems",
        "Feeling lonely"
    ]]])

    raw_pred = depression_model.predict(features)[0]
    final_label = dep_encoder.inverse_transform([raw_pred])[0]

    return jsonify({"prediction": final_label})


# ================================
# EMOTION KEYWORD PREDICTOR
# ================================
@app.route("/ai_predict", methods=["POST"])
def ai_predict():
    data = request.get_json()
    full_text = " ".join(data.get("answers", [])).lower()

    anxiety_words = ["anxious", "worry", "panic", "nervous", "tense", "yes"]
    depression_words = ["sad", "depressed", "hopeless", "cry", "empty", "worthless", "yes"]
    stress_words = ["tired", "stressed", "pressure", "overwhelmed", "yes"]

    if any(w in full_text for w in anxiety_words):
        return jsonify({"prediction": "Anxiety Signs", "confidence": 0.85})

    if any(w in full_text for w in depression_words):
        return jsonify({"prediction": "Depression Signs", "confidence": 0.87})

    if any(w in full_text for w in stress_words):
        return jsonify({"prediction": "Stress Signs", "confidence": 0.80})

    return jsonify({"prediction": "Stable Mood", "confidence": 0.70})

from flask import send_from_directory

@app.route("/")
def home():
    return send_from_directory(".", "senior.html")

# ================================
# RUN FLASK APP
# ================================
if __name__ == "__main__":
    app.run(debug=True)
