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

interface = gr.Interface(
     theme=gr.themes.Glass(),
    css=css

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
    css = """
/* Animated Gradient Background */
body{
    background: linear-gradient(-45deg,#0f172a,#1e3a8a,#0f766e,#2563eb);
    background-size:400% 400%;
    animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG{
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}

/* Glass Card */
.gradio-container{
    max-width:900px !important;
    margin:30px auto !important;
    border-radius:20px;
    background:rgba(255,255,255,.12) !important;
    backdrop-filter:blur(15px);
    box-shadow:0 10px 40px rgba(0,0,0,.4);
    animation:fadeIn 1.2s ease;
}

/* Fade Animation */
@keyframes fadeIn{
    from{
        opacity:0;
        transform:translateY(30px);
    }
    to{
        opacity:1;
        transform:translateY(0);
    }
}

/* Title Animation */
h1{
    color:white !important;
    text-align:center;
    animation:glow 2s infinite alternate;
}

@keyframes glow{
    from{
        text-shadow:0 0 5px #60a5fa;
    }
    to{
        text-shadow:0 0 20px #38bdf8;
    }
}

/* Inputs */
input{
    border-radius:12px !important;
    transition:.3s;
}

input:focus{
    transform:scale(1.03);
    border:2px solid #3b82f6 !important;
}

/* Button */
button{
    background:linear-gradient(90deg,#2563eb,#06b6d4)!important;
    color:white!important;
    border:none!important;
    border-radius:12px!important;
    transition:.3s;
}

button:hover{
    transform:scale(1.05);
    box-shadow:0 0 20px #38bdf8;
}
)
# ------------------------------------------

if __name__ == "__main__":
    # Render network configuration
    interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
