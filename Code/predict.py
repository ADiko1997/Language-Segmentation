from argparse import ArgumentParser
import tensorflow
import keras as k 
import preprocessing as p 
from keras.models import model_from_json
#import model as m
import numpy as np 
from keras.preprocessing.sequence import pad_sequences  
from keras.models import load_model
import pickle 
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):
    #1st step import vocabularie from Resources/vocab_uni\bi.pkl
    with open(resources_path +'vocab_bi.pkl', 'rb') as f:
        vocab_bi = pickle.load(f)
    with open(resources_path +'vocab_uni.pkl', 'rb') as f:
        vocab_uni = pickle.load(f)
    #Preproces the file given as input for prediction
    pred_x_uni, pred_x_bi, pred_y,max_len = p.preprocessing_predict(vocab_uni, vocab_bi,path=input_path)
    #loads the model from the file
    model = load_model(resources_path +'final_model.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    print("Loaded model from disk")
    print('Starting to make predictions \n')
    #makes the prediction and transforms the predicted file into BIES format and outputs it
    # to the given output path
    prediction = model.predict([pred_x_uni, pred_x_bi])
    text_file = []
    sen=[]
    for row in prediction:       
        line=[]
        for element in row:
            val=np.argmax(element)
            if val==0:
                    line.extend("B")
            elif val==1:
                line.extend("I")
            elif val==2:
                line.extend("E")
            else:
                line.extend("S")
        sentence=''.join(line)
        sen.append(sentence)
        text_file.append(sentence)
    text = p.join_file(input_path)
    
    for i in range(len(text)):
        text_file[i] = text_file[i][:len(text[i])]
    
    file = open(output_path,'w')
    
    for i in text_file:
        file.write(i)
        file.write('\n')
    file.close()
    print('Ended with success')
    return 

if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)