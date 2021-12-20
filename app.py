import pickle
import streamlit as st
from helper import *
import numpy as np
from pathlib import Path
import base64
import spacy
#from spacy import displacy
import spacy_streamlit
# loading the trained model
formal_pickle_in = open('formal.pkl', 'rb') 
formal_classifier = pickle.load(formal_pickle_in)
casual_pickle_in = open('casual.pkl', 'rb')
casual_classifier = pickle.load(casual_pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(str_in, mode):   
 
    # Pre-processing user input    
    df_predict = turn2correctdf(nltk.pos_tag(word_tokenize(str_in)))

    if (mode == 0):
        model = formal_classifier
        prediction = model.predict(turndf2modelInput(df_predict))
        for i in range(df_predict.index.size):
            df_predict["Tag"][i] = prediction[0][i]
        prediction2 = model.predict(turndf2modelInput(df_predict))
        for i in range(df_predict.index.size):
            df_predict["Tag"][i] = prediction[0][i]    
        #st.dataframe(df_predict)    

    else:
        model = casual_classifier
        prediction = model.predict(turndf2modelInput_casual(df_predict))
        for i in range(df_predict.index.size):
            df_predict["Tag"][i] = prediction[0][i]
        prediction2 = model.predict(turndf2modelInput_casual(df_predict))
        for i in range(df_predict.index.size):
            df_predict["Tag"][i] = prediction[0][i]    
        #st.dataframe(df_predict)    
    
    df_predict = df_predict.drop(df_predict.columns[0], axis=1)
    return prediction, df_predict

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
  

# this is the main function in which we define our webpage  
#def main():       
    # front end elements of the web page 
html_temp = """ 
<div style ="background-color:yellow;padding:13px"> 
<h1 style ="color:black;text-align:center;">NER ML App</h1> 
</div> 
"""
reason = """Why there are two types of sentence?
\nBecause during model training, I discover that word capitalization is THE most important factor that influence whether a word is a 
named entity, which makes a lot of sense since all names are capitalized. Therefore, I trained two models, one for formal, grammatically correct
sentence, and one for casual sentence that doesn't care about the capitalization aspect of each word"""
st.markdown(html_temp, unsafe_allow_html = True) 
st.sidebar.markdown('''[<img src='data:image/png;base64,{}' class='img-fluid' width=32 height=32>](https://streamlit.io/)'''.format(img_to_bytes("logo.png")), unsafe_allow_html=True)
st.sidebar.header('NER Simple System')
#st.caption("Check the side bar for spacy implementation")
navigation=st.sidebar.selectbox('Navigation',['My own model','Spacy model','About this'])
if navigation == "My own model":
    st.caption("Check the side bar for spacy implementation")
    str_input = st.text_input("Sentence for NER checking") 
    st.caption("Sample sentences:  \n James went to Paris on December 12 to see Harry Potter movie and went home in Ho Chi Minh City at midnight.")
    st.caption("The German firm works as a sub-contractor for Shell.")
    st.caption("Her opponent was billionaire Sebastian Pinera, the candidate of a rightist alliance.")
    setence_type = st.selectbox("Sentence type", ("Formal","Casual"))
    st.caption(reason)
    if st.button("Predict"): 
        if (len(str_input)!= 0):
            if (setence_type == "Formal"):
                result, df = prediction(str_in=str_input, mode= 0) 
            else:
                result, df = prediction(str_in=str_input, mode= 1)
            #for i in range(len(result[0])):
                #print(result[0][i])
            st.dataframe(df)
            st.success("Result is: {}".format(result[0]))
        else: 
            st.error('Please input a string')

if navigation == "Spacy model":
    st.caption("Check the side bar for my own implementation")
    st.caption('**Hi! This is an advanced version of the NER system that support beautiful visualization from spacy-streamlit, unlike the previous two option that is from my own training model without spacy implementation.**')
    models = ["en_core_web_sm"]
    default_text =  "James went to Paris on December 12 to see Harry Potter movie and went home in Ho Chi Minh City at midnight."
    spacy_streamlit.visualize(models, default_text)

if navigation == "About this":
    st.markdown("")
    st.markdown("This is a web app for the bonus point assigment for the **Machine Learning** course. Notebooks for generating the models are included in the **Github** repository, please check it out :D.")
    st.markdown("*https://github.com/qdgiang/ner-system*")

    
    #if __name__=='__main__': 
        #main()