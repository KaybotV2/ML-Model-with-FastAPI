import os
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File
from mangum import Mangum
import numpy as np
import pandas as pd
import io
import random
from sklearn.utils import resample
from scipy.signal import resample as scipy_resample
from sklearn.metrics import classification_report
from tensorflow import keras
from keras.models import load_model
import boto3

app = FastAPI()
handler = Mangum(app)

# S3 bucket configuration
S3_BUCKET = 'ml-model-bucket-22092024'
K_MODEL_KEY ='model/model.h5'

# Initialize the S3 client
s3 = boto3.client('s3')

# List objects in the S3 bucket (for debugging)
def list_s3_objects(bucket):
    try:
        response = s3.list_objects_v2(Bucket=bucket)
        if 'Contents' in response:
            for obj in response['Contents']:
                print(obj['Key'])
        else:
            print("No objects found in the bucket.")
    except Exception as e:
        print(f"Error listing objects in S3 bucket: {e}")

list_s3_objects(S3_BUCKET)

# Utility function to download a file from S3
def download_file_from_s3(bucket, key):
    file_stream = io.BytesIO()
    try:
        s3.download_fileobj(bucket, key, file_stream)
        file_stream.seek(0)  # Reset stream position
    except Exception as e:
        raise RuntimeError(f"Error downloading {key} from S3: {e}")
    return file_stream

def load_h5_model_from_s3(bucket, key):
    file_stream = download_file_from_s3(bucket, key)
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        temp_file.write(file_stream.read())
        temp_file.flush()
        
        # Load the model using Keras
        return load_model(temp_file.name)

# Load the model from S3
try:
    model_h5 = load_h5_model_from_s3(S3_BUCKET, K_MODEL_KEY)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.get("/")
def index():
    return {"message": "API is running"}

# Utility functions for signal manipulation
def add_gaussian_noise(signal, mean=0, std=0.05):
    noise = np.random.normal(mean, std, signal.shape)
    return signal + noise

def stretch(signal):
    l = int(signal.shape[0] * (1 + (random.random() - 0.5) / 3))
    y = scipy_resample(signal, l)
    if l < signal.shape[0]:
        padded = np.zeros(signal.shape)
        padded[:l] = y
        return padded
    return y[:signal.shape[0]]

def amplify(signal):
    alpha = (random.random() - 0.5)
    return signal * (1 + alpha)

def add_amplify_and_stretch_noise(signal):
    if random.random() < 0.33:
        return stretch(signal)
    elif random.random() < 0.66:
        return amplify(signal)
    return amplify(stretch(signal))

# Resample Dataset
def resample_dataset(df, labels):
    random_seed = 42
    class_indexes = {label: df.index[labels == label].tolist() for label in labels.unique()}
    
    resampled_indexes = {
        label: resample(indexes, replace=True, n_samples=10000, random_state=random_seed)
        for label, indexes in class_indexes.items()
    }
    
    resampled_data = pd.concat([df.iloc[resampled_indexes[label]] for label in resampled_indexes], axis=0)
    
    return resampled_data

# Prediction API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the training CSV contents
    contents = await file.read()
    dataset = pd.read_csv(io.BytesIO(contents), header=None)
    
    # Assuming the last column contains the labels
    labels = dataset.iloc[:, -1].astype('category')
    obs = np.array(dataset.iloc[:, :-1])  # All but last column

    if dataset.shape[1] != 188:  # 187 features + 1 label
        return {"error": "Dataset must contain exactly 188 columns."}

    # Resample dataset for balanced class distribution
    resampled_df = resample_dataset(dataset, labels)

    # Make predictions using both models
    resampled_signals = resampled_df.iloc[:, :-1].to_numpy()
    true_labels = resampled_df.iloc[:, -1].to_numpy()
    
    predictions_h5 = model_h5.predict(resampled_signals)


    # Get predicted classes
    predicted_classes_h5 = np.argmax(predictions_h5, axis=1)
  

    # Calculate metrics for both models
    report_h5 = classification_report(true_labels, predicted_classes_h5, output_dict=True)

    # Prepare the response
    response = {
        "Test Accuracy (Overall)": {
            "model_h5": report_h5["accuracy"],
        },
        "Detailed Metrics": {
            "model_h5": report_h5,
        }
    }

    return response

if __name__ == "__main__":
    port = 8000
    print(f"Running the FastAPI server on port {port}.")
    uvicorn.run("app:app", host="0.0.0.0", port=port)
