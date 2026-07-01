# test_agent.py
import httpx

API_URL = "http://127.0.0.1:8000/chat"

def simulate_full_conversation():
    # Maintain a stateless list of messages to pass back and forth
    history = []
    
    # ==========================================
    # TURN 1: Sending Vague Request
    # ==========================================
    print("\n--- Turn 1: Sending Vague Request ---")
    history.append({"role": "user", "content": "Hi, I need an assessment test for my team."})
    
    response = httpx.post(API_URL, json={"messages": history}, timeout=30.0)
    if response.status_code != 200:
        print(f"Failed at Turn 1: {response.text}")
        return
        
    data = response.json()
    print(f"AI Reply: {data['reply']}")
    print(f"Recommendations: {data['recommendations']}")
    print(f"End of Conversation: {data['end_of_conversation']}")
    
    # Save the assistant's reply to history for the next turn
    history.append({"role": "assistant", "content": data['reply']})
    
    # ==========================================
    # TURN 2: Providing Specific Details
    # ==========================================
    print("\n--- Turn 2: Providing Specific Details ---")
    # Feed specific target role and parameters based on the catalog
    history.append({
        "role": "user", 
        "content": "They are entry-level software engineers. I want to evaluate their coding skills and Python logic."
    })
    
    response = httpx.post(API_URL, json={"messages": history}, timeout=30.0)
    if response.status_code != 200:
        print(f"Failed at Turn 2: {response.text}")
        return
        
    data = response.json()
    print(f"AI Reply: {data['reply']}")
    print(f"Recommendations: {data['recommendations']}")
    print(f"End of Conversation: {data['end_of_conversation']}")

if __name__ == "__main__":
    simulate_full_conversation()