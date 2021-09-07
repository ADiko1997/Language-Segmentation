import preprocessing as p
import pandas as pd 
from keras.preprocessing.sequence import pad_sequences  
import keras as k
import numpy as np

#Making ready all the data needed for training the model and evaluating it
concat_vocab, vocab_uni, vocab_bi, text = p.preprocessing_basic()
labels_to_num , y_set = p.preprocessing_label_transformation(path='./union 1.utf8')
train_x_uni, train_x_bi, train_y = p.preprocessing_training(text, vocab_uni, vocab_bi, labels_to_num)
dev_x_uni, dev_x_bi, dev_y = p.preprocessing_test(vocab_uni, vocab_bi,path='./union.utf8')
test_x_uni, test_x_bi, test_y = p.preprocessing_test(vocab_uni, vocab_bi,path='./msr_test_gold.utf8')

#Batch generator to avoid memory problems, Paul`s code
def batch_generator(X,X2, Y, batch_size, shuffle=False):
    if not shuffle:
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            yield [X[start:end],X2[start:end]], Y[start:end]
    else:
        perm = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            yield [X[perm[start:end]],X2[perm[start:end]] ], Y[perm[start:end]]
#Reference for creating this model is keras functional api 
def create_keras_model(vocab_size1,vocab_size2,  hidden_size):
    print("Creating KERAS model")
    model = k.models.Sequential()
    #Creating the first embeding layer for unigrams
    un=k.layers.Input(shape=(None,),name='unigrams')
    em_unigram = k.layers.Embedding(vocab_size1, 64, mask_zero=True)(un)
    
    #Creating the second embedding layer for bigrams    
    bi=k.layers.Input(shape=(None,), name='bigrams')
    em_bigram = k.layers.Embedding(vocab_size2, 32, mask_zero=True)(bi)
    #merging the two embeding layers
    merged= k.layers.concatenate([em_unigram, em_bigram], axis=-1)
    #Creating the mode, a stcked biLstm model
    lstm_result = k.layers.LSTM(hidden_size,dropout = 0.3, recurrent_dropout=0.25, return_sequences=True)(merged)
    lstm_result1 =k.layers.LSTM(hidden_size,dropout=0.3 ,recurrent_dropout=0.25, return_sequences=True)(lstm_result)
    output  = k.layers.TimeDistributed(k.layers.Dense(4, activation='softmax'))(lstm_result1)
    model = k.models.Model(inputs=[un, bi], outputs=output)
    optimizerA = k.optimizers.Adam(0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizerA, metrics=['acc'])

    return model
     


# running the model
batch_size = 128
epochs = 0
model = create_keras_model(len(vocab_uni),len(vocab_bi)+len(vocab_uni), 264)
model.load_weights('./model_new.h5')
model.summary()
print("\nStarting training...")
model.fit_generator(batch_generator(train_x_uni,train_x_bi, train_y, 128, True ),steps_per_epoch=3000,epochs=epochs,
          shuffle=True,validation_data=([dev_x_uni,dev_x_bi],dev_y))
score= model.evaluate([test_x_uni, test_x_bi], test_y)
score1=model.evaluate([dev_x_uni, dev_x_bi],dev_y)
print(score, score1)
print("Evaluation completed. \n")
#saving model
model.save('final_model.h5')
# # serialize weights to HDF5
#model.save_weights("./model_new1.h5")
print("Saved model to disk")

# if __name__ == "__main__":
#     import sys
#     model(int(sys.argv[1]))    