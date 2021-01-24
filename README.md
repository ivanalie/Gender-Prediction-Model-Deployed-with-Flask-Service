## Gender Prediction ML Model and Deployment using a Flask Service
This project demonstrates training and serving a machine learning model to assign a gender (male, female) to first names.

## Folder content
Important files to this project folder
1. train_model.py - This code is for training ML model to predict gender based on first name. Model is trained using 'name_gender.csv' dataset.
2. best_nn_model.h5 (not in present folder) - Run train_model.py to generate this model. The accuracy of this model is 0.92 on validation data.
3. requirements.txt - Package requirements for this project.
4. flask_app.py - This has the flask APIs that receives a name as input through API call and return the gender prediction produced by the trained model.
5. request.py - This is for user to call the APIs and request the predicted values. A few sample names are included. Run this after flask_app is up and running.
6. Dockerfile - This is the dockerfile to build a docker image containing all the dependencies required for this project.
7. docker-compose.yml - This docker configuration file defines the service requirements.
8. name_gender.csv - This is the dataset used for model training.
9. predictname.json - This is a json file with sample first name. User can also make the API call to request the predicted values by passing a json file.

## Training ML model
Generate trained model (best_nn_model.h5) by running train_model.py. 
```
python train_model.py
```
If there is a need to train with different parameters, use the following command and change the optional param to a different value
e.g. changing batch-size to 126, nepochs to 50, dropout to 0.3, ndims to 128
```
python train_model.py --train-file <path to csv of training dataset> --batch-size 256 --ndims 512 --nepochs 30 --dropout 0.2 --model-path <path to save model>
```

## Installation/ Deploying and serving the ML model
Ensure docker is running, and go to this project directory to build a docker image with the following command:
```
docker build --tag predict-gender-docker .
docker run --publish 5000:5000 predict-gender-docker
```
Or
```
docker-compose up --build
```

Once the docker image is built and running, the serving app is ready to use. Try one of these three ways to get gender prediction results with first names:
1. Run the following command
```
python request.py 
```
Or <br>
2. On browser, navigate to URL http://127.0.0.1:5000
<br> Or <br>
3. Run the following command (by specifying path to the json file)
```
curl -X POST -H "Content-Type: application/json" -d @predictname.json http://localhost:5000/predict_json 
```
