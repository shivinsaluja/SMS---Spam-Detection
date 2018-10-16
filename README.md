# SMS - Spam Detection

The objective of this project is to develop an algorithm which can detect whether a given SMS is a spam SMS or not. The algorithm makes use of a Recurrent Neural Network model which is implemented using LSTM's. It is a classification problem solved using the aforementioned algorithm. The model is implemented using a Bi-Directional LSTM with the dimension of the hidden layer as 300 nodes.


# Approach


## Data Cleaning and Processing

The dataset contains a collection of Spam as well as Normal text messages. Each message/text has been labled as Spam or not spam using labels 0 and 1. 
0 - Spam SMS
1 - Not a Spam SMS.

The dataset is cleaned and special symbols and characters are removed using various libararies in python. Certain short forms of different words are replaced with their proper forms  for e.g shouldnt is changed to should not and i'll is changed to I will.Stop words and common words are removed from all the messages using Parts of Speech (Pos) tagging and capital letter are converted to small letters. Each message/text is converted to tokens and is furthur converted into vectors (word embeddings) of dimension 384.Padding is applied to each message so that each message/text is of same length. 


## Training 

The training is done using a Bi-directional LSTM network with hidden dimension as 300 nodes. The output layer of the network is a dense layer with softmax activation function applied on it. Early stopping has also been implemented to decrease the training time. The model configeration is saved in model.json and weights are saved in model.h5

# Steps to Run

1. The Dataset should be in the form of a csv file with 2 columns namely 'Text' and 'Label'. 
2. Run the file 'Spam_Detection.py' to perform spam detection. 

# Role of different Functions - 

clean_dataset() - The function clean dataset removes different symbols from the text. It also removes extra spaces and replaces certain words for e.g - 'Couldnt is changed to could not' etc. 

filter_word() - The function is used to filter words and only return words which are present in the POS_SET.

Pad_vec_sequences() - The function is used to pad the vector sequences which have length less than the maximum length with extra spaces. 

remove_special_characters() - This function is used to remove the special symbols,characters and words from the text. 

train_dataset() - This function is used to train our model on the preprocessed dataset. The model used here is a Bidirectional LSTM.

Test_model() - This function is used to load the model and predict the output to a particular text. We give the 'Text/SMS' as the input to the model to predict whether a SMS is a spam one or not.


