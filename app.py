import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
# Load the tokenizers
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('tag_tokenizer.pkl', 'rb') as f:
    tag_tokenizer = pickle.load(f)

model = load_model('ner_model.keras')

max_length = 34  # Update this value if different

def predict_ner(sentence):
    # Preprocess the input sentence
    input_sequence = tokenizer.texts_to_sequences([sentence])
    input_padded = pad_sequences(input_sequence, maxlen=max_length, padding="post")

    # Make predictions
    predictions = model.predict(input_padded)

    # Get the predicted NER tags
    prediction_ner = np.argmax(predictions, axis=-1)

    # Convert predictions to NER tags
    NER_tags = [tag_tokenizer.index_word.get(num, 'O') for num in list(prediction_ner.flatten())]

    # Split the sentence into words
    words = sentence.split()

    # Return words with their corresponding NER tags
    return list(zip(words, NER_tags[:len(words)]))

# Streamlit UI
st.title("Named Entity Recognition (NER) with RNN")

st.write("Enter a sentence to predict the named entities:")

# Input text box
sentence = st.text_input("Sentence")

if st.button("Predict"):
    if sentence:
        results = predict_ner(sentence)
        
        st.write("Predicted Named Entities:")
        for word, tag in results:
            st.write(f"{word}: {tag}")
    else:
        st.write("Please enter a sentence to get predictions.")


