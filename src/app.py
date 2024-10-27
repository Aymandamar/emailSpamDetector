import streamlit as st
import pickle
import spacy


# Load the spaCy model
nlp = spacy.load('en_core_web_sm')



#headline in the app hhh
st.header("""SMS Spam Detection Classifier""")
st.write("""This app predicts the **Spam** of an SMS! using different machine learning models""")
# Load the model 
list_model=[ 'SVC','KN','NB','DT','LR','RF','AdaBoost','GDBT','xgb','mnb','gnb']
model_name=st.sidebar.selectbox('Select Model',list_model)

def load_model(model_name):
    model = pickle.load(open(f'src/pkl_models/{model_name}.pkl', 'rb'))
    return model

model=load_model(model_name)

def preprocess_text(text):
    doc = nlp(text.lower())  # Convertir le texte en minuscule
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)


# Create a text box for user input
user_input = st.text_input("Enter the SMS text")


# Preprocess the user input
user_input_processed = preprocess_text(user_input)
#show the processed text
st.write(f"Processed text: {user_input_processed}")


#immport the vectorizer
tfidf = pickle.load(open('src/vectorizer.pkl','rb'))



# When the 'Predict' button is clicked, make the prediction and store it
if st.button("Predict"):
    vector_input = tfidf.transform([user_input_processed]).toarray()  # Convert sparse to dense
    prediction = model.predict(vector_input)[0]
    # Display the result
    st.write(f"Prediction: {prediction}")
    if prediction == 1:
        st.header("Le sms est un Spam")
    elif prediction == 0:
        st.header("Le sms n'est pas un Spam")
    else:
        st.header("Erreur de prediction")

