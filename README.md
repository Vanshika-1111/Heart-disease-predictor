🩺 Heart Disease Predictor

This project utilizes a Machine Learning (ML) model to predict the presence of heart disease based on 13 key health parameters. This is a full-stack application consisting of a Python (Flask) backend and an HTML/CSS/JavaScript frontend.

⚙️ Core Project Structure

Heart.csv: The primary dataset used for training.

train_new.py: The script responsible for data cleaning, preprocessing (using Yeo-Johnson transformation), training the Random Forest model, and saving the model artifacts.

app.py: The Flask API backend that loads the trained model and serves the prediction endpoint (/api/predict) and the frontend HTML.

index.html: The responsive frontend form that collects user data and communicates with the backend.

model_artifacts/: Directory created by train_new.py containing the saved model files (.pkl).

🚀 Setup and Running the Application

1. Environment Setup

It is highly recommended to use a virtual environment (venv).

# 1. Create and activate the virtual environment
python -m venv venv
.\venv\Scripts\activate


2. Install Dependencies

Use the requirements.txt file to install all necessary Python libraries:

pip install -r requirements.txt


3. Train the Model

Run the training script to prepare the model artifacts. This must be run successfully before starting the server.

python train_new.py


4. Start the Backend Server

Start the Flask server. This server will run on port 8080 and automatically serve the prediction API and the frontend HTML page.

python app.py


5. Access the Frontend

Once the terminal shows * Running on http://127.0.0.1:8080, open your web browser and navigate directly to:

[http://127.0.0.1:8080/](http://127.0.0.1:8080/)


You can now use the form to get real-time predictions from your model.