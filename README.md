# ECG Heartbeat Categorization with Machine Learning
This project utilizes a machine learning model to analyze and categorize ECG heartbeat data. The model is trained on a heartbeat categorization dataset and is deployed on an AWS serverless architecture using FastAPI.

# Key Features:
- **Machine Learning Model:** Processes and categorizes ECG heartbeat data.
- **Dataset:** Utilizes a dedicated ECG heartbeat categorization dataset for training.
- **Deployment:** Implements a serverless architecture on AWS for efficient and scalable deployment with FastAPI.

**Important note:** I do not own the ECG Heartbeat ML Model; it is derived from a source on Kaggle, the owner of the model is George Youssef: https://www.kaggle.com/code/georgeyoussef1/ecg-heartbeat-random-forest-classifier

## Setup Python environment for ML

### Step 1: Install Anaconda
Visit [conda webite](https://www.anaconda.com/), locate and download the free version of Anaconda for your operating system. Then install it like you do for any other software

### Step 2: Install Python
Visit [Python org](https://www.python.org/downloads/)

### Step 3: Activate the environment
```bash
conda activate mlenv
```
### Step 4: Install the packages
For this project we will need pandas, scikit-learn and matplotlib.

```bash
pip install boto3 uvicorn mangum pydantic fastapi python-multipart tensorflow pyyaml h5py numpy pandas scikit-learn scipy
```

### Step 5: Install Jupyter lab and start
See [Installation](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)

```bash
pip install jupyterlab

# Then, start jupyterlab.
jupyter lab
```
## Deactivate the environment

```bash
conda deactivate
```

## Using Docker Image

### Build and Test the Image Locally

Update .env File with AWS Credentials. This will be used by Docker for when we want to test the image locally.

```bash

AWS_ACCESS_KEY_ID=XXXXX
AWS_SECRET_ACCESS_KEY=XXXXX
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=s3-bucket-name

```

These commands can be run from image/ directory to build, test, and serve the app locally.

```bash

docker build -t ecg_model_lambda-image .

```

```bash

docker run --rm -it \
    --entrypoint python \
    --env-file .env \
    ecg_model_lambda-image app.py

```

## Running Locally as a Server

Assuming you've build the image from the previous step.

```bash 

docker run --rm -p 8000:8000 \
    --entrypoint python \
    --env-file .env \
    ecg_model_lambda-image app.py

```    

Then go to http://0.0.0.0:8000/docs to try it out.

![FastApi screen 1](/ecg_model_api1.png)
![FastApi screen 2](/ecg_model_api2.png)
![FastApi screen 3](/ecg_model_api3.png)
![FastApi screen 4](/ecg_model_api4.png)

## Deploy to AWS

I have put all the AWS CDK files into aws-resources/. Go into the folder and install the Node dependencies.

### prerequisite 

You need to have an AWS account, AWS CDK and AWS CLI set up on your machine. 


```bash

npm install

#then

cdk deploy

```

