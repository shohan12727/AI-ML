import streamlit as st 

st.title("Input your files(image)",anchor=False)
st.divider()

st.image("images/devteam.jpg")

st.divider()

images = st.file_uploader("Enter your image",
                          type=['jpg','jpeg','png'],
                          accept_multiple_files=True
                          )
print(type(images))

if images:
    if(len(images)>2):
        st.warning("You uploaded 3 photos")
    col = st.columns(len(images))   
    
    
    for i, per_image in enumerate(images):
        with col[i]:
            st.image(per_image)  








