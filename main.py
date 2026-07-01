import os
import json
from fastapi import FastAPI, HTTPException
from typing import List
from catalog_fetch import CatalogFetch
from pydantic import BaseModel
from google import genai
from google.genai import types


app = FastAPI()

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"

catalog_fetch = CatalogFetch(CATALOG_URL)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str
    description: str = ""

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[RecommendationItem]
    end_of_conversation: bool

@app.get("/health")
def health_check():
    if len(catalog_fetch.catalog) > 0:
        return {"status": "ok"}
    return{"status": "loading"}

SYSTEM_PROMPT_TEMPLATE = """You are the official SHL Assessment Recommender bot. Your sole objective is to help users find ideal assessment products from our strict catalog context.

=== CATALOG CONTEXT ===
{catalog_context}

=== CONVERSATION PROTOCOL ===
1. Clarification Phase: If the request is vague, keep recommendations empty [], set end_of_conversation to false, and ask targeted clarifying questions.
2. Recommendation Phase: Once you have enough specifics, provide the recommendations from the CATALOG CONTEXT. However, keep `end_of_conversation` as FALSE to allow the user to ask questions, compare products, or adjust their criteria.
3. Zero Hallucination: Populated items must match the EXACT Name and EXACT URL found in the CATALOG CONTEXT.
4. Conversation Termination: Set `end_of_conversation` to TRUE *only* when the user explicitly confirms, says "Confirmed", "Perfect", "Sounds good", or clearly signals they are done and satisfied with the recommendations.

=== OUTPUT FORMAT ===
You must respond strictly according to the following JSON schema layout:
{{
  "reply": "Your conversational text response or comparisons go here.",
  "recommendations": [
    {{
      "name": "Exact Name from context",
      "url": "Exact URL from context",
      "test_type": "The clean Type flag (P, C, or K) from context",
      "description": "A brief summary of why it matches"
    }}
  ],
  "end_of_conversation": true or false
}}
"""
@app.post("/chat", response_model = ChatResponse)
def chat_endpoint(request: ChatRequest):
    try: 
        catalog_context = catalog_fetch.get_assessment_data()
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(catalog_context=catalog_context) 

        contents =[]
        for msg in request.messages:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents = contents,
            config = types.GenerateContentConfig(
                system_instruction = system_prompt,
                temperature=0.0,
                response_mime_type = "application/json",
                response_schema = ChatResponse,
            ),
        )

        clean_text = response.text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.lstrip("`").replace("json", "", 1).strip()
            clean_text = clean_text.rstrip("`").strip()
        return ChatResponse.model_validate_json(clean_text)
    
    except Exception as e:
        print(f"Backend Crash Log: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    