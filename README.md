# SMS - Spam Detection

1. The Dataset should be in the form of a csv file with 2 columns namely 'Text' and 'Label'. 
2. Run the file 'Spam_Detection.py' to perform spam detection. 

Role of different Functions - 

clean_dataset() - The function clean dataset removes different symbols from the text. It also removes extra spaces and replaces certain words for e.g - 'Couldnt is changed to could not' etc. 

filter_word() - The function is used to filter words and only return words which are present in the POS_SET.

Pad_vec_sequences() - The function is used to pad the vector sequences which have length less than the maximum length with extra spaces. 

remove_special_characters() - This function is used to remove the special symbols,characters and words from the text. 

train_dataset() - This function is used to train our model on the preprocessed dataset. The model used here is a Bidirectional LSTM.

Test_model() - This function is used to load the model and predict the output to a particular text. We give the 'Text/SMS' as the input to the model to predict whether a SMS is a spam one or not.


