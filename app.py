import pickle
import streamlit as st
from helper import *
import numpy as np
 
# loading the trained model
pickle_in = open('first.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(str_in):   
 
    # Pre-processing user input    
 
    # Making predictions 
    prediction = classifier.predict(fromString2modelInput(str_in))
    return prediction
    #if prediction == 0:
        #pred = 'Rejected'
    #else:
        #pred = 'Approved'
    #return pred
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">NER ML App</h1> 
    </div> 
    """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 

    navigation=st.sidebar.selectbox('Navigation',['Home','About me'])




    # display the front end aspect
    #st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    #Gender = st.selectbox('Gender',("Male","Female"))
    #Married = st.selectbox('Marital Status',("Unmarried","Married")) 
    str_input = st.text_input("Sentence for NER checking") 
    #LoanAmount = st.number_input("Total loan amount")
    #Credit_History = st.selectbox('Credit_History',("Unclear Debts","No Unclear Debts"))
    #result =""
    setence_type = st.selectbox("Sentence type", ("Casual","Formal"))
    st.write("Hmm")
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(str_in=str_input) 
        for i in range(len(result[0])):
            print(result[0][i])
        st.success("result is: {}".format(result))
        #st.success('Your loan is {}'.format(result))
        #print(LoanAmount)
     
if __name__=='__main__': 
    main()