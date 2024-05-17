import time #Iwish
import os
import json
import requests
import streamlit as st
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import google.generativeai as genai


def generate_facebook_post(business_type, target_audience, post_goal, post_tone, include, avoid):
    """
    Generates a Facebook post prompt for an LLM based on user input.

    Args:
        business_type: The type of business, e.g., fashion retailer, fitness coach.
        target_audience: A description of the target audience.
        post_goal: The goal of the Facebook post.
        post_tone: The desired tone of the post.
        include: Elements to include in the post (e.g., images, videos, links).
        avoid: Elements to avoid in the post (e.g., long paragraphs, technical jargon).

    Returns:
        A string containing the LLM prompt.
    """
    prompt = f"""I am a {business_type}.

    Please help me write a detailed Facebook post that will engage my target audience, {target_audience}.

    Here are some additional details to consider:

    * **Post Goal:** {post_goal}
    * **Post Tone:** {post_tone}
    * **Include:** {include}
    * **Avoid:** {avoid}

    **Example Post Structure:**

    1. **Attention-Grabbing Opening:** Start with a question or a bold statement to capture attention.
    2. **Engaging Content:** Briefly describe the main message or offer, highlighting key benefits or features.
    3. **Call-to-Action (CTA):** Encourage the audience to take a specific action (e.g., visit a link, comment, share).
    4. **Multimedia:** Mention the types of multimedia elements to include (e.g., images, videos).
    5. **Hashtags:** Include relevant hashtags to increase post visibility.

    """
    try:
        response = generate_text_with_exception_handling(prompt)
        return response
    except Exception as err:
        st.error(f"An error occurred while generating the prompt: {e}")
        return None


def main():
    st.title("Alwrity - Facebook Post Generator")
    st.markdown("This app will help you create a Facebook post prompt for an LLM.")

    try:
        # Collect user inputs with default values
        business_type = st.text_input("**What is your business type?**", placeholder="fitness coach")
        target_audience = st.text_input("**Describe your target audience:**", placeholder="fitness enthusiasts")
        post_goal = st.selectbox("**What is the goal of your post?**", ["Promote a new product", "Share valuable content", "Increase engagement", "Other"], index=2)
        post_tone = st.selectbox("**What tone do you want to use?**", ["Informative", "Humorous", "Inspirational", "Upbeat", "Casual"], index=3)
        include = st.text_input("**What elements do you want to include?** (e.g., images, videos, links, hashtags, questions)", placeholder="short video with a sneak peek of the challenge")
        avoid = st.text_input("**What elements do you want to avoid?** (e.g., long paragraphs, technical jargon)", placeholder="long paragraphs")

        if st.button("Write FaceBook Post"):
            if not business_type or not target_audience:
                st.error("🚫 Provide required inputs. Least, you can do..")
            
            generated_post = generate_facebook_post(business_type, target_audience, post_goal, post_tone, include, avoid)
            if generated_post:
                st.subheader(f'**🧕 Verify: Alwrity can make mistakes.**')
                st.write("## Generated Facebook Post:")
                st.write(generated_post)
                st.write("\n\n\n\n\n")
            else:
                st.error("Error: Failed to generate Facebook Post.")
    except Exception as e:
        st.error(f"An error occurred: {e}")



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text_with_exception_handling(prompt):
    """
    Generates text using the Gemini model with exception handling.

    Args:
        api_key (str): Your Google Generative AI API key.
        prompt (str): The prompt for text generation.

    Returns:
        str: The generated text.
    """

    try:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 8192,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]

        model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        convo = model.start_chat(history=[])
        convo.send_message(prompt)
        return convo.last.text

    except Exception as e:
        st.exception(f"An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    main()
