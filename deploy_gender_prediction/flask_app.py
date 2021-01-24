import numpy as np
import pandas as pd
import string
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

def generate_ohe_matrix(char_list,char_to_idx):
    '''Create one hot encoding matrix for all characters
    :param char_list (list) list of character to encode
    :param char_to_idx(dict) dictionary mapping character to index  
    :return: np.array of one hot encoding
    '''
    arr_init = np.zeros(len(char_list))
    tmp_dic = dict()
    for char in (char_list):
        arr_init[char_to_idx[char]] = 1
        tmp_dic[char_to_idx[char]] = arr_init
        arr_init = np.zeros(len(char_list))
    return tmp_dic

def name_to_idx_padded(name):
    '''Convert name to padded index value
    :param name (str) first name in string format
    :return: list of index representation of name, length of the list
    '''
    name = name.lower()
    # pad name
    if (len(name) < padded_len):
        padded_name = name + '$' * (padded_len - len(name))
    else:
        padded_name = name[:padded_len]
    
    padded_name_in_idx = [char_to_idx[char] for idx, char in enumerate(padded_name) if idx < padded_len] 
    
    return padded_name_in_idx, len(padded_name_in_idx)

def name_to_ohe(name):
    '''Convert name to one hot encoding values
    :param name (str) first name in string format
    :return: numpy array with one hot encoding represetation of name 
    '''
    tmp = []
    pad_name_idx, _ = name_to_idx_padded(name)
    for idx in pad_name_idx:
        tmp.append(char_to_ohe_matrix[idx])
    return np.asarray(tmp)

# Initialize variables
char_list = [char for char in string.ascii_lowercase] + [ '$']
len_char = len(char_list)
char_to_idx = {char:idx for idx, char in enumerate(char_list)}
idx_to_char = {idx:char for idx, char in enumerate(char_list)}
padded_len = 20
char_to_ohe_matrix = generate_ohe_matrix(char_list, char_to_idx)


home_message = 'Gender Prediction! Try predict with http://localhost:5000/predict?name=[putfirstname] '
app = Flask(__name__)

@app.route('/')
def home():
    '''
    Print a home message
    '''
    return home_message

@app.route('/predict')
def predict_api():
    ''' For user to get the prediction result using html browser
    '''
    name =  request.args.get("name")
    name_ohe = name_to_ohe(str(name).lower()) # convert name to ohe
    prediction = (nn_model.predict(np.asarray([name_ohe])) > 0.7)[0][0]
    result = 'Female' if prediction == False else('Male' if prediction == True else None)
    return '{} is a {}'.format(name, result)

@app.route('/predict_json', methods=['POST'])
def predict_api_json():
    '''For user to get the prediction result using API call request 
    '''
    jsondata = request.json
    name = jsondata['name']
    name_ohe = name_to_ohe(str(name).lower()) # convert name to ohe
    prediction = (nn_model.predict(np.asarray([name_ohe])) > 0.7)[0][0]
    result = 'Female' if prediction == False else('Male' if prediction == True else None)
    return jsonify({name: result})


if __name__ == "__main__":
    filename = './best_nn_model.h5'
    nn_model = load_model(filename)
    app.run(debug=False,  host='0.0.0.0')