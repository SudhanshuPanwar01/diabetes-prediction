# app.py

import os
import gradio as gr
import joblib
# Load the trained Decision Tree model at startup
deployed_dt = joblib.load('diabetes_prediction_model.pkl')

# --- CODE BLOCK: PREDICTION LOGIC FOR 5 FEATURES ---
def predict_diabetes(pregnancies, glucose, insulin, bmi, age):
    # The model expects a 2D array matching the exact order of x_train
    input_data = [[pregnancies, glucose, insulin, bmi, age]]
    prediction = deployed_dt.predict(input_data)
    
    # Interpret the binary outcome (typically 1 for positive, 0 for negative)
    if prediction[0] == 1:
        return "Prediction: High Risk of Diabetes (Positive)"
    else:
        return "Prediction: Low Risk of Diabetes (Negative)"
# ---------------------------------------------------

# --- CODE BLOCK: GRADIO INTERFACE SETUP ---
css = """
body {
    background: #f4f8fb;
}

.gradio-container {
    background: #f4f8fb;
}

h3 {
    color: #0d47a1;
    text-align: center;
}

.info-box {
    background: #e3f2fd;
    border: 2px solid #1976d2;
    border-radius: 10px;
    padding: 15px;
    color: #000;
    font-size: 16px;
}
"""

interface = gr.Interface(

    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies (Number of times pregnant)"),
        gr.Number(label="Glucose (Plasma glucose concentration)"),
        gr.Number(label="Insulin (2-Hour serum insulin)"),
        gr.Number(label="BMI (Body mass index)"),
        gr.Number(label="Age (Years)")
    ],
    outputs=gr.Text(label="Assessment Result"),
    title="Diabetes Prediction System",
    description="""
    <h3>Diabetes Prediction System</h3>

<p>Enter the medical metrics below to predict diabetes risk using a Decision Tree Machine Learning model.</p>

<hr>

<p>
<b>👨‍💻 Developed By:</b> Sudhanshu Panwar<br>
<b>📞 Contact:</b> +91-9416407198<br>
<b>📧 Gmail:</b> sudhanshupanwar0111@gmail.com<br>
<b>🏫 College:</b> Panipat Institute of Engineering and Technology (PIET)
</p>
"""
)
# ------------------------------------------

if __name__ == "__main__":
    # Render network configuration
    interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
