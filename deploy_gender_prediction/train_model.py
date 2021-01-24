import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Embedding, Dense, Activation, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train gender prediction from name')
    parser.add_argument('--train-file', default='name_gender.csv', help='File of training data (default: name_gender.csv)')
    parser.add_argument('--batch-size', default=256, type=int, help='Batch size for training (default: 256)')
    parser.add_argument('--ndims', default=512, type=int, help='Number of dimensions for NN layers (default: 512)')
    parser.add_argument('--nepochs', default=30, type=int, help='Number of epochs to train (default: 30)')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout rate (default: 0.2)')
    parser.add_argument('--model-path', default='best_nn_model.h5', help='Path to save model (default: best_nn_model.h5)')
    args = parser.parse_args()


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

    # Initialize one hot encoding for characater list
    char_to_ohe_matrix = generate_ohe_matrix(char_list, char_to_idx)

    # Read training dataset
    df = pd.read_csv(args.train_file,  header=None, names=['name','gender','base_proba'])
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'M' else (0 if x == 'F' else None))
    df['name_length'] = df['name'].apply(lambda x: len(x))
    names = df.name.apply(lambda x: str(x).lower())
    labels = df.gender
    padded_len = df['name_length'].max() + 5; #print (padded_len)

    # Train test split
    name_train, name_test, label_train, label_test = train_test_split(names, labels, test_size=0.2, stratify=labels)
    x_train = np.asarray([name_to_ohe(name) for name in name_train])
    y_train = np.asarray(label_train)
    x_test = np.asarray([name_to_ohe(name) for name in name_test])
    y_test = np.asarray(label_test)

    # Build Model
    nn_model = Sequential()
    nn_model.add(Bidirectional(LSTM(args.ndims, return_sequences=True), backward_layer=LSTM(args.ndims, return_sequences=True, go_backwards=True), input_shape=(padded_len,len_char)))
    nn_model.add(Dropout(args.dropout))
    nn_model.add(Bidirectional(LSTM(args.ndims)))
    nn_model.add(Dropout(args.dropout))
    nn_model.add(Dense(1, activity_regularizer=l2(0.002)))
    nn_model.add(Activation('sigmoid'))

    nn_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    # Fit and save best model
    batch_size = 256
    callback = EarlyStopping(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint('best_nn_model.h5', monitor='val_loss', mode='min', verbose=1)
    reduce_lr_acc = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, verbose=1, min_delta=1e-4, mode='max')
    nn_model.fit(x_train, y_train, validation_data =(x_test, y_test), 
              batch_size=args.batch_size, epochs=args.nepochs, verbose=1,
              callbacks=[callback, checkpoint, reduce_lr_acc])
  
  
if __name__ == "__main__":
    main()