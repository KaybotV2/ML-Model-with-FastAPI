# Use an appropriate base image with Python and necessary libraries
FROM public.ecr.aws/lambda/python:3.8

# Copy the requirements file and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
# Install the specified packages
RUN pip install -r requirements.txt --upgrade

# For local testing.
EXPOSE 8000

# Set IS_USING_IMAGE_RUNTIME Environment Variable
ENV IS_USING_IMAGE_RUNTIME=True

# Command for AWS Lambda to find the handler inside src/app.py
COPY src/app.py ${LAMBDA_TASK_ROOT}
