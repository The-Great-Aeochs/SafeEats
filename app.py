import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import streamlit as st
import tempfile

# Functions as defined before

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.write(f"Error processing image: {e}")
        return ""

@st.cache_resource
def load_llama_model(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def analyze_ingredients(text, health_conditions, tokenizer, model):
    prompt = f"""
    Given the following list of ingredients, identify any ingredients that are potentially harmful for the following health conditions: {', '.join(health_conditions)}.

    Ingredients:
    {text}

    List the harmful ingredients and provide a brief explanation for each.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=500, num_return_sequences=1, temperature=0.2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def parse_model_response(response):
    flagged_ingredients = []
    pattern = r"\d+\.\s*Ingredient:\s*(.+?)\s*-\s*(.+)"
    matches = re.findall(pattern, response, re.DOTALL)
    for match in matches:
        ingredient, explanation = match
        flagged_ingredients.append({
            'ingredient': ingredient.strip(),
            'explanation': explanation.strip()
        })
    return flagged_ingredients

# Streamlit app code
def main():
    st.title("Ingredient Label Analyzer")
    st.write("Upload an ingredient label image to identify potentially harmful ingredients based on your health preferences.")

    st.sidebar.header("User Preferences")
    health_conditions = st.sidebar.multiselect(
        "Select your health conditions:",
        options=["Diabetes", "Hypertension", "High Cholesterol", "Celiac Disease", "Lactose Intolerance"]
    )

    if not health_conditions:
        st.sidebar.warning("Please select at least one health condition to analyze ingredients.")

    uploaded_file = st.file_uploader("Upload Ingredient Label Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Image uploaded successfully.")
        
        # Temporary file for image processing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            image.save(temp_path)
        
        # Adding an "Analyze" button
        if st.button("Analyze Ingredients"):
            # Extract text from the image
            text = extract_text_from_image(temp_path)
            st.write("Extracted Text:")
            st.write(text)
            
            # Load model and analyze ingredients
            tokenizer, model = load_llama_model("gpt2")  # Replace "gpt2" with the appropriate model
            
            if text and health_conditions:
                response = analyze_ingredients(text, health_conditions, tokenizer, model)
                flagged_ingredients = parse_model_response(response)
                
                # Display results
                st.write("Flagged Ingredients:")
                if flagged_ingredients:
                    for item in flagged_ingredients:
                        st.write(f"- **Ingredient:** {item['ingredient']}")
                        st.write(f"  **Explanation:** {item['explanation']}")
                else:
                    st.write("No harmful ingredients found based on selected health conditions.")

if __name__ == "__main__":
    main()
