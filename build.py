import re

def autocorrect(input_text):
    """
    Comprehensive autocorrection system for simple conversations
    """
    # Convert input to title case for initial processing
    input_text = input_text.title()
    
    # Static phrases and their corrections
    static_phrases = {
        "Thank You": "Thank you",
        "I Love You": "I love you",
        "Yes": "Yes",
        "No": "No",
        "Good Meet You": "Nice to meet you",
        "Hi Good Meet You": "Hi, nice to meet you",
        "Hello Good Meet You": "Hello, nice to meet you",
        "Me Student": "I am a student",
        "You Student": "Are you a student?",
    }
    
    # Check for exact static phrases first
    if input_text in static_phrases:
        return static_phrases[input_text]
    
    # Question patterns (check these first)
    question_patterns = {
        "What You Name": "What is your name?",
        "Who You Mother": "Who is your mother?",
        "Who You Father": "Who is your father?",
        "Where You Live": "Where do you live?",
        "How You": "How are you?",
    }
    
    if input_text in question_patterns:
        return question_patterns[input_text]
    
    # Greeting patterns
    greeting_patterns = {
        r'^(Hi|Hello)$': lambda m: f"{m.group(1)}!",
        r'^(Hi|Hello)\s+How\s+You$': lambda m: f"{m.group(1)}, how are you?",
        r'^(Hi|Hello)\s+([A-Z][a-zA-Z]*)$': lambda m: f"{m.group(1)} {m.group(2)}!",
    }
    
    # Check greeting patterns
    for pattern, replacement in greeting_patterns.items():
        match = re.match(pattern, input_text)
        if match:
            return replacement(match) if callable(replacement) else replacement
    
    # Main conversation patterns
    patterns = {
        # Family patterns (check these before personal info patterns)
        r'^Me\s+(Mother|Father)$': lambda m: f"My {m.group(1).lower()}",
        r'^You\s+(Mother|Father)$': lambda m: f"Your {m.group(1).lower()}",
        r'^(Mother|Father)$': lambda m: f"My {m.group(1).lower()}",
        
        # Number handling
        r'^Me\s+([0-9]+)$': lambda m: f"I am {m.group(1)} years old",
        
        # Personal information patterns
        r'^Me\s+Name\s+([A-Z][a-zA-Z]*)$': lambda m: f"My name is {m.group(1)}",
        r'^Me\s+([A-Z][a-zA-Z]*)$': lambda m: f"I am {m.group(1)}",
        r'^Me\s+Live\s+([A-Z][a-zA-Z\s]*)$': lambda m: f"I live in {m.group(1)}",
        
        # Feeling/state patterns
        r'^Me\s+Feel\s+(Good|Fine|Great|Beautiful)$': lambda m: f"I feel {m.group(1).lower()}",
        r'^Me\s+(Good|Fine|Great|Beautiful)$': lambda m: f"I am {m.group(1).lower()}",
        r'^You\s+(Good|Fine|Great|Beautiful)$': lambda m: f"Are you {m.group(1).lower()}?",
        r'^Me\s+Hungry$': lambda m: "I am hungry",
        r'^You\s+Hungry$': lambda m: "Are you hungry?",
        
        # Question patterns with remaining words
        r'^(Who|What|Why|When|How)\s+You\s+(.+)$': lambda m: f"{m.group(1)} is your {m.group(2).lower()}?",
        r'^(Who|What|Why|When|How)\s+Me\s+(.+)$': lambda m: f"{m.group(1)} is my {m.group(2).lower()}?",
        
        # Meet patterns
        r'^Meet\s+([A-Z][a-zA-Z\s]*)$': lambda m: f"Nice to meet you, {m.group(1)}",
        r'^You\s+Meet\s+([A-Z][a-zA-Z\s]*)$': lambda m: f"Have you met {m.group(1)}?",
    }
    
    # Try each pattern and return the first match
    for pattern, replacement in patterns.items():
        match = re.match(pattern, input_text)
        if match:
            return replacement(match) if callable(replacement) else replacement
    
    # If no pattern matches, return the original text
    return input_text

# Interactive mode
print("\nInteractive mode (type 'exit' to quit):")
while True:
    user_input = input("\nEnter text to autocorrect: ")
    if user_input.lower() == 'exit':
        break
    print("Corrected:", autocorrect(user_input))