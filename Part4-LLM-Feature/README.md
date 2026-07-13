# Part 4: LLM-Powered Feature — Model Prediction Explanation Pipeline

**Chosen Track:** Track C — Model Prediction Explanation Pipeline

## 1. Prompt Design Decisions

### System Prompt (Verbatim)
```text
You are a highly analytical AI assistant specializing in explainable machine learning models for FinTech systems. 
Your task is to explain a model's classification prediction given its input features, the predicted class, and the prediction probability.

RULES:
1. Output ONLY a valid JSON object. No explanations, no markdown formatting (do not wrap in ```json), and no surrounding conversational text.
2. The JSON output must strictly conform to the target schema.
3. Be professional and objective in your reasons.
4. Top reasons must directly link feature values to the prediction (e.g., "high value of days_to_forward indicates administrative delay").
```

### User Prompt Template
```text
Explain the following machine learning model prediction:

--- INPUT DATA ---
Features: {features}
Predicted Class: {predicted_class} (where 1 = Extreme Delay, 0 = Normal Speed)
Prediction Probability: {probability:.4f}

--- SCHEMA REQUIREMENT ---
Return a JSON object containing exactly these fields:
{{
    "prediction_label": "string: 'Extreme Delay' or 'Normal Speed'",
    "confidence_level": "string: 'low' | 'medium' | 'high'",
    "top_reason": "string: primary feature driving this prediction",
    "second_reason": "string: secondary feature supporting this prediction",
    "next_step": "string: recommended operational next step based on the prediction"
}}
```

### Temperature Choice Justification
We chose **`temperature=0.0`** for our core pipeline. In LLM generation, temperature controls the randomness of the model's token selection. A temperature of 0.0 makes the model strictly deterministic, always selecting the highest-probability next token. This is critical for structured data and JSON extraction tasks because any variability (high temperature) increases the risk of formatting errors, schema deviations, or JSON syntax violations, which would crash our validation pipeline.

---

## 2. Temperature A/B Comparison

| Input Scenario | Output at `temperature=0.0` | Output at `temperature=0.7` | Key Difference |
|---|---|---|---|
| **High Delay Predict** | `{"prediction_label": "Extreme Delay", "confidence_level": "high", "top_reason": "days_to_forward is high", ...}` | `{"prediction_label": "Extreme Delay", "confidence_level": "high", "top_reason": "The days_to_forward value is elevated...", ...}` | At `temp=0.7`, the wording becomes more verbose and varied. At `temp=0.0`, the keys and values are concise, highly consistent, and fit the JSON schema exactly. |

---

## 3. PII Guardrail Test Results
Before every LLM API call, the input query passes through a regular expression check to detect any emails or 10-digit phone numbers:
- **Clean Input Test:** *"Customer complaint regarding a credit card charge of $50.00."*
  - *Result:* **PASSED (Proceeded to LLM)**
- **PII Input Test:** *"My email is test@company.com and my phone number is 123-456-7890. I was double-billed."*
  - *Result:* **BLOCKED (PII Detected, call aborted)**

---

## 4. End-to-End Pipeline Demonstration

| Input Feature Vector | Predicted Class | Probability | Explanation JSON | Validation Status | Pass / Block |
|---|---|---|---|---|---|
| **Input 1:** Web submit, short narrative | 0 | 0.024 | `{"prediction_label": "Normal Speed", ...}` | **PASS** | **Pass (No PII)** |
| **Input 2:** Mail submit, long narrative | 1 | 0.812 | `{"prediction_label": "Extreme Delay", ...}` | **PASS** | **Pass (No PII)** |
| **Input 3:** Phone submit, PII included | - | - | `None` | **FAIL (Blocked)** | **Blocked (PII)** |

---

## 5. Adding a Screen Recording of the Flask App Dashboard
To display a live demo of the Flask Dashboard in this README, follow these steps:
1. Record a 30-second screen capture of your local browser interface navigating the Flask app.
2. Save or convert the video file into a `.gif` format (e.g., using tool like FFmpeg or free web converters).
3. Save the resulting file as `assets/dashboard_demo.gif` in this repository directory.
4. Add the markdown image link below to embed it:
   
   ```markdown
   ![Dashboard Demo](assets/dashboard_demo.gif)
   ```
5. If using an MP4 video, you can embed it natively using HTML5 video tags in this markdown document:
   ```html
   <video src="assets/dashboard_demo.mp4" controls width="100%"></video>
   ```
