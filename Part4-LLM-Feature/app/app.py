import os
import re
import json
import requests
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from jsonschema import validate, ValidationError

app = Flask(__name__)

# Load model from Part 3 (looking up one directory level)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../Part3-Advanced-Modeling/best_model.pkl')
if not os.path.exists(MODEL_PATH):
    # Fallback to local app directory
    MODEL_PATH = 'best_model.pkl'

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load best_model.pkl. Details: {e}")
    model = None

# Expected output schema for JSON validation
EXPLANATION_SCHEMA = {
    "type": "object",
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"],
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"}
    }
}

# PII Guardrail
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))

# Call LLM via Requests
def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    api_key = os.environ.get('LLM_API_KEY')
    if not api_key:
        return json.dumps({
            "prediction_label": "Error",
            "confidence_level": "low",
            "top_reason": "API Key is missing in environment variables.",
            "second_reason": "N/A",
            "next_step": "Please configure LLM_API_KEY."
        })
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"API Error: Status {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        print(f"Network error calling LLM: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model file best_model.pkl not found."}), 500
        
    data = request.get_json()
    
    # Extract features matching the model training schema
    raw_features = {
        'narrative_length': float(data.get('narrative_length', 100)),
        'company_response_length': float(data.get('company_response_length', 50)),
        'ZIP code numeric': float(data.get('zip_code', 0)),
        'Submitted via_encoded': float(data.get('submitted_via', 1))
    }
    
    # Include product dummy variables (One-Hot Encoded variables default to 0)
    product = data.get('product', 'Credit card')
    for col in model.feature_names_in_:
        if col not in raw_features:
            # Map one-hot encoded product column
            if col.startswith('Product_') and col == f'Product_{product}':
                raw_features[col] = 1
            else:
                raw_features[col] = 0
                
    # Create input DataFrame
    input_df = pd.DataFrame([raw_features])
    
    # Run predictions
    pred = int(model.predict(input_df)[0])
    prob = float(model.predict_proba(input_df)[0][pred])
    
    # Extract free text for explanation
    narrative_text = data.get('narrative', '')
    
    # Guardrail Check
    if has_pii(narrative_text):
        return jsonify({
            "prediction": pred,
            "probability": prob,
            "guardrail_status": "Blocked",
            "explanation": {
                "prediction_label": "Blocked",
                "confidence_level": "low",
                "top_reason": "PII Guardrail triggered.",
                "second_reason": "Input contained email or phone number.",
                "next_step": "Remove PII and submit again."
            }
        })
        
    # Prompt Setup
    system_prompt = (
        "You are a highly analytical AI assistant explaining ML predictions. "
        "Output ONLY a valid JSON object matching the target schema. "
        "No markdown, no conversation, no markdown wrapper codeblocks."
    )
    user_prompt = f"""
    Explain this model prediction:
    Features: {raw_features}
    Predicted Class: {pred} (1=Extreme Delay, 0=Normal Speed)
    Probability: {prob:.4f}
    
    JSON Schema:
    {{
        "prediction_label": "string",
        "confidence_level": "low" | "medium" | "high",
        "top_reason": "string",
        "second_reason": "string",
        "next_step": "string"
    }}
    """
    
    # Call LLM
    raw_llm_response = call_llm(system_prompt, user_prompt, temperature=0.0)
    
    # Parse and Validate Output
    explanation = None
    if raw_llm_response:
        try:
            cleaned_response = raw_llm_response.strip()
            # Handle potential markdown code wrappers in responses
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:-3].strip()
                
            explanation = json.loads(cleaned_response)
            validate(instance=explanation, schema=EXPLANATION_SCHEMA)
            validation_status = "Pass"
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Validation failed: {e}")
            validation_status = "Fail"
            # Fallback output
            explanation = {
                "prediction_label": "Extreme Delay" if pred == 1 else "Normal Speed",
                "confidence_level": "medium",
                "top_reason": f"Fallback: High value of processing features.",
                "second_reason": f"Prediction probability is {prob:.2%}",
                "next_step": "Manual triage by operations team."
            }
    else:
        validation_status = "Failed API Call"
        explanation = {
            "prediction_label": "Extreme Delay" if pred == 1 else "Normal Speed",
            "confidence_level": "low",
            "top_reason": "LLM API failed to respond.",
            "second_reason": "N/A",
            "next_step": "Verify API configurations."
        }
        
    return jsonify({
        "prediction": pred,
        "probability": prob,
        "guardrail_status": "Pass",
        "validation_status": validation_status,
        "explanation": explanation
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
