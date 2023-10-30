import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


# Load the trained model
def load_model(model_filename):
    model = pickle.load(open(model_filename, 'rb'))
    print(model)
    return model

# Data Cleaning for "Claim Description" column
def clean_text(text):
    # Remove special characters, numbers, and extra whitespaces
    text = ' '.join(word for word in text.split() if word.isalpha())
    # Convert text to lowercase
    text = text.lower()
    return text

# Preprocess the input data (similar to previous steps)
def preprocess_data(input_data):
    tfidf_vectorizer = TfidfVectorizer(max_features=50,min_df=1,stop_words='english')    
    input_data['Claim Description'] = input_data['Claim Description'].apply(clean_text)
    input_data.dropna(subset=['Claim Description'], inplace=True)       
    X_tfidf = tfidf_vectorizer.fit_transform(input_data['Claim Description'])    
    return X_tfidf

# Make predictions using the loaded model
def predict(input_data,preprocessed_data):

    # Load the trained model  
    model1 = load_model('xgbModel_cc.pkl') # Specify the trained model file 
    predictions_cc = model1.predict(preprocessed_data)
    
    model2 = load_model('xgbModel_as.pkl') # Specify the trained model file
    predictions_as = model2.predict(preprocessed_data)

    data = pd.read_excel('Dataset_Public.xlsx')
    # Encode Categorical Target Variables
    label_encoder_coverage = LabelEncoder()
    label_encoder_accident = LabelEncoder()

    data['Coverage Code'] = label_encoder_coverage.fit_transform(data['Coverage Code'])
    data['Accident Source'] = label_encoder_accident.fit_transform(data['Accident Source'])

    # Decode the predicted values back to their original labels
    predictions = pd.DataFrame({
        'Claim Description':input_data['Claim Description'],
        'Coverage Code': label_encoder_coverage.inverse_transform(predictions_cc),
        'Accident Source': label_encoder_accident.inverse_transform(predictions_as)
    },index=list(range(len(input_data['Claim Description']))))


    return predictions

# Save the predictions to an Excel file
def save_predictions_to_excel(predictions, output_filename):
    output_data = pd.DataFrame({'Predicted Coverage Code': predictions})
    output_data.to_excel(output_filename, index=False)

# Main Streamlit app
def main():
    st.title('Claim Prediction App')
    # Define the sidebar content
    # st.sidebar.title("Sidebar Title")
    

    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx'])
    button = st.sidebar.button("Run")

    if button and uploaded_file is not None:        

        # Load the input Excel file
        input_data = pd.read_excel(uploaded_file)        
        
        # Preprocess the data (cleaning and TF-IDF vectorization)
        preprocessed_data = preprocess_data(input_data)
        
        # Make predictions
        predictions = predict(input_data,preprocessed_data)

        # Display a success alert
        st.success("Predictions excel file downloaded successfully with name output_predictions ")
        
        # Create a placeholder for the four columns
        
        col1, col2, col3, col4 = st.columns(4,gap="small")
        with col1:
            st.subheader("Coverage Code Precision")
            st.subheader("64.76")
        with col2:
            st.subheader("Coverage Code Recall")
            st.subheader("65.62")
        with col3:
            st.subheader("Accident Source Precision")
            st.subheader("60.93")
        with col4:
            st.subheader("Accident Source Recall")
            st.subheader("56.82")

        # Use st.write() to display the DataFrame as a table
        st.write(predictions)     
        

        # Save the predictions to an output Excel file
        output_filename = f'output_predictions.xlsx'
        # save_predictions_to_excel(predictions, output_filename, index=False)
        predictions.to_excel(output_filename, index=False)


        # Download link for the output predictions file
        st.markdown(f"Download the predictions [here](output_predictions.xlsx)")

if __name__ == '__main__':
    main()
