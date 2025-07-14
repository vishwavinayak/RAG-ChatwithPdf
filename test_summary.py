#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from llm_config import get_groq_llm

# Load environment variables
load_dotenv()

def test_summarization():
    """Test the summarization functionality"""
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found in environment variables.")
        return
    
    # Initialize LLM
    print("🔄 Loading LLM...")
    llm = get_groq_llm()
    
    # Test text (first part of your PDF)
    test_text = """
    Science 134Light – Reflection and Refraction9 CHAPTER We see a variety of objects in the world ar ound us. However , we ar e unable to see anything in a dark room. On lighting up the room, things become visible. What makes things visible? During the day, the sunlight helps us to see objects. An object reflects light that falls on it. This reflected light, when received by our eyes, enables us to see things. We are able to see thr ough a transpar ent medium as light is transmitted through it. There are a number of common wonderful phenomena associated with light such as image formation by mirrors, the twinkling of stars, the beautiful colours of a rainbow, bending of light by a medium and so on. A study of the properties of light helps us to explore them.
    """
    
    # Create a simple summarization prompt
    prompt = f"""
    You are an expert summarizer. Create a comprehensive summary of the provided text.
    
    Guidelines:
    - Identify the main topics and key concepts
    - Include important definitions and explanations
    - Structure the summary with clear sections
    - Use bullet points for key points
    - Make it informative and educational
    
    Text to summarize:
    {test_text}
    
    Summary:
    """
    
    print("🔄 Generating summary...")
    try:
        response = llm.invoke(prompt)
        
        print(f"✅ Response type: {type(response)}")
        print(f"✅ Response has content attr: {hasattr(response, 'content')}")
        
        if hasattr(response, 'content'):
            summary = response.content.strip()
            print(f"✅ Summary length: {len(summary)} characters")
            print("\n" + "="*50)
            print("SUMMARY:")
            print("="*50)
            print(summary)
            print("="*50)
        else:
            print("❌ Response doesn't have content attribute")
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_summarization() 