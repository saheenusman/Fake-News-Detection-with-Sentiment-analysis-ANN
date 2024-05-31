import streamlit as st
import numpy as np
import joblib
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import nltk
import tensorflow as tf
import base64

# Download NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')

def get_base64_of_binary_file(bin_file):
  """
  Reads the contents of a binary file and returns its base64-encoded representation.
  """
  with open(bin_file, 'rb') as f:
    data = f.read()
  return base64.b64encode(data).decode()


def set_background(img_file):
  """
  Sets the background image for the Streamlit app using base64 encoding.

  Args:
      img_file (str): Path to the local image file.
  """
  page_bg_img = '''
  <style>
  body {
      background-image: url("data:image/png;base64,%s");
      background-size: cover;  # Adjust background size as needed (cover, contain, etc.)
  }
  </style>
  ''' % get_base64_of_binary_file(img_file)
  st.markdown(page_bg_img, unsafe_allow_html=True)




# Load the pre-trained model
@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Load TF-IDF vectorizer and MaxAbsScaler
@st.cache_resource
def load_vectorizer_and_scaler(vectorizer_path, scaler_path):
    vectorizer = joblib.load(vectorizer_path)
    scaler = joblib.load(scaler_path)
    return vectorizer, scaler

vectorizer, scaler = load_vectorizer_and_scaler('tfidf_vectorizer.joblib', 'maxabs_scaler.joblib')

# Preprocessing function
def preprocess_text(text):
    tk = TweetTokenizer()
    tokens = tk.tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z\s]', '', token) for token in tokens]
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(token.lower()) for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    processed_text = ' '.join(tokens)
    return processed_text

def predict(model, input_text):
    processed_text = preprocess_text(input_text)
    # Apply TF-IDF vectorization
    x_tfidf = vectorizer.transform([processed_text]).toarray()
    
    # Apply MaxAbsScaler
    x_scaled = scaler.transform(x_tfidf)
    
    # Reshape input to match the shape of the model
    expected_input_shape = model.input_shape
    reshaped_input = np.reshape(x_scaled, (1, *expected_input_shape[1:]))
    y_pred = model.predict(reshaped_input)[0]
    prediction = np.where(y_pred > 0.5, 1, 0)
    return prediction

def main():
    # Load the model
    model_path = 'fake_news_detection_model_simple.h5'  # Update with your model path
    model = load_model(model_path)
    set_background('back.png')

    #Title and custom CSS for dark theme
    st.markdown(
        """
        <style>
      
        .copyright {
        position: fixed;
        bottom: 0;
        right: 0;
        padding: 10px;
        font-size: 20px;
        color: white;
        }
        </style>
        <div class="copyright"><span style="font-family: Trebuchet MS;"> BY SAHEEN USMAN M</div>
        """,
        unsafe_allow_html=True
    )
    st.title('Fake News Detection')

    # Initialize session state for the input text
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ''

    # Text input
    st.session_state.input_text = st.text_area('Enter the news text:', st.session_state.input_text)

    # Make predictions
    if st.button('Predict'):
        if st.session_state.input_text.strip() == '':
            st.error('Please enter some text.')
        else:
            prediction = predict(model, st.session_state.input_text)
            if prediction == 0:
                st.success('The news is real.')
            else:
                st.error('The news is fake.')

if __name__ == '__main__':
    main()
