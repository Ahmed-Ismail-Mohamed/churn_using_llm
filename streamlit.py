import streamlit as st
import pickle
from langchain.prompts import PromptTemplate
import pandas as pd
from io import StringIO
from langchain_google_genai import ChatGoogleGenerativeAI

google_api_key = "AIzaSyDCvYe_Gc_7uAQw-OQHZZmL52bPT67MKbM"
llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=google_api_key,temperature=0)

with open('clf.pkl', 'rb') as file:
    clf = pickle.load(file)
with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)
with open('scl.pkl', 'rb') as file:
    scl = pickle.load(file)

prompt = """
You are a Bank churn assistant named Zenith Arabia, your job is to take the user question and extract the following information from the user's question

credit_score          int64
country              object
gender               object
age                   int64
tenure                int64
balance             float64
products_number       int64
credit_card           int64
active_member         int64
estimated_salary    float64

Examples:
credit_score,country,gender,age,tenure,balance,products_number,credit_card,active_member,estimated_salary
619,France,Female,42,2,0.00,1,1,1,101348.88
608,Spain,Female,41,1,83807.86,1,0,1,112542.58
502,France,Female,42,8,159660.80,3,1,0,113931.57
699,France,Female,39,1,0.00,2,0,0,93826.63
850,Spain,Female,43,2,125510.82,1,1,1,79084.10


Note:
    - The output must be in csv format that is able to be converted to dataframe if all information are presented
    - the output must be in the format as the given examples
    - return the csv format only without any additional text
    - if any of the values are not presented in the question 
        1. do not return the csv format, just return a normal sentence
        2. apologies to the user and tell him a good sentence about the missing information

example for the output if all information are presented:
credit_score,country,gender,age,tenure,balance,products_number,credit_card,active_member,estimated_salary
619,France,Female,42,2,0.00,1,1,1,101348.88
----------------------
credit_score,country,gender,age,tenure,balance,products_number,credit_card,active_member,estimated_salary
850,Spain,Female,43,2,125510.82,1,1,1,79084.10

Question: {user_input}
"""

Prompt_Template = PromptTemplate(
    input_variables=["user_input"],
    template=prompt
)

def get_answer(question):
    prompt_filled_by_question =Prompt_Template.format(
        user_input = question,
        )
    res = llm.invoke(prompt_filled_by_question).content
    print(res)
    data_io = StringIO(res)
    df_sample = pd.read_csv(data_io)
    if df_sample.empty:
        return res
    else:
        encoded_data = encoder.transform(df_sample[['gender', 'country']])
        encoded_columns = encoder.get_feature_names_out()
        encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoded_columns)
        df_enc = pd.concat([df_sample.drop(['gender', 'country'], axis=1), encoded_df], axis=1)
        data_scl = scl.transform(df_enc)
        prediction = clf.predict(data_scl)
        if prediction[0] == 0:
            return 'The customer will not exit (0)'
        else:
            return 'The customer will exit (1)'

# Streamlit app
st.title("Question Answering with OpenAI GPT-3")

question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question:
        answer = get_answer(question)
        st.write("Answer:")
        st.write(answer)
    else:
        st.write("Please enter a question.")

