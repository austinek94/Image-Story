
### 0) Load necessary libraries
from dotenv import find_dotenv, load_dotenv
from distutils.core import setup
import transformers
from transformers import pipeline
import requests
import os
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

### 1) Convert image to text

def imagetotext(url):
    image_to_text = pipeline('image-to-text',model = 'Salesforce/blip-image-captioning-base')
    
    text = image_to_text(url)[0]['generated_text']
    
    print(text)
    
    return text

### 2) Convert text to text story

tokenizer = transformers.AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
model = transformers.AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")

def generate_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids,
        max_length=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response_ids = outputs[0]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text

### 3) Convert text story to audio
def text_to_speech(text):
    
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer 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"}

    payloads = {
        'inputs': text
    }

    response =  requests.post(API_URL,headers=headers,json=payloads)
    with open('audio.flac','wb') as file:
        file.write(response.content)

### 4) Set up UI and main
def main():
    st.set_page_config(page_title='Image Story')
    
    st.header('Generate a story from an image')
    upload_image = st.file_uploader('Choose an image',type='jpg')
    
    if upload_image is not None:
        bytes_data = upload_image.getvalue()
        with open(upload_image.name,'wb') as file:
            file.write(bytes_data)
        st.image(upload_image,caption='Uploaded Image',use_column_width=True)
        prompt = 'Tell a story about '+imagetotext(upload_image.name)
        single_turn_prompt = f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
        text_removal = f"GPT4 Correct User: {prompt} GPT4 Correct Assistant:"
        response_text = generate_response(single_turn_prompt)[len(text_removal):]
        text_to_speech(response_text)
        
        with st.expander('scenario'):
            st.write(prompt)
        with st.expander('story'):
            st.write(response_text)
        
        st.audio('audio.flac')
        
if __name__ == '__main__':
    main()