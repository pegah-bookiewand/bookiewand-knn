# Bookiewand AI
Currently, the AI service consists of two applications:
1. **GST Validation**
2. **Account Mapping**

These services are machine learning processes for validating the GST tax names and 
predicting the account codes for journal entries. 
These services use a K-Nearest Neighbors (KNN) approach to predict the correct 
tax names and account codes based on similar journal line entries for a given tenant ID.

## Setup and Deployment
Prerequisites are:
- Python 3.10
- Docker and Docker Compose

Note that in case the codes are executed on the local machine, LocalStack must be installed and its Docker
container must be up and running:
```shell
localstack start -d
```

The service can be deployed using Docker Compose:
```shell
# 1. Deploy only the model selection service
docker compose up --build -d gst_validation_model_selection
docker compose up --build -d account_mapping_model_selection

# 2. Deploy only the training service
docker compose up --build -d gst_validation_training
docker compose up --build -d account_mapping_training

# 3. Optional: Deploy only the inference API
docker compose up --build -d gst_validation_inference
docker compose up --build -d account_mapping_inference
```

## Workflow
1. Run the `<application>_model_selection` to find the best features and model hyperparameters.
The results will be saved to S3, and `<application>_training` automatically loads the best configurations. 
2. Run `<application>_training` to train the corresponding model. Note that if `<application>_model_selection` is not
executed beforehand, the `<application>_training` will run with default feature set and model configurations.
3. Either use the `<application>` function defined in `ai/<application>/src/api`, or run the `<application>_inference`
docker to set up the FastAPI server, corresponding to the `<application>` function.

## Environment Configuration
The example file `.env.example` can be used to create a `.env` file in the `bookiewand/ai/`:


## Contact
[amirhossein.nouranizadeh@gmail.com](mailto:amirhossein.nouranizadeh@gmail.com)