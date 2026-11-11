import streamlit as st 

st.title("Input your infomation", anchor = False)
st.divider()

name = st.text_input("Enter you name")


st.write("Your name is: ", name)

st.divider()

age = st.number_input("Enter your number", placeholder="Type your age")
st.write("Your age is: ",age)

pressed = st.button("Enter to confirm")

if pressed:
    st.write(f"Your name is {name} and your age is {age}")