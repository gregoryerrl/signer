import re

def autocorrect(input_text):
    """
    Comprehensive autocorrection system for daily conversations
    """
    # Convert input to title case for initial processing
    input_text = input_text.title()
    
    # Static phrases and their corrections
    static_phrases = {
        # Greetings and Farewells
        "Hi": "Hi!",
        "Hello": "Hello!",
        "Good Morning": "Good morning!",
        "Good Afternoon": "Good afternoon!",
        "Good Evening": "Good evening!",
        "Good Night": "Good night!",
        "Bye": "Goodbye!",
        "Bye Bye": "Goodbye!",
        "See You": "See you later!",
        "See You Later": "See you later!",
        "Good Bye": "Goodbye!",
        
        # Common Expressions
        "Thank You": "Thank you",
        "Thanks": "Thank you",
        "Thank": "Thank you",
        "Please": "Please",
        "Sorry": "I'm sorry",
        "Excuse": "Excuse me",
        "Excuse Me": "Excuse me",
        "Welcome": "You're welcome",
        "No Problem": "No problem",
        "Ok": "Okay",
        "Yes": "Yes",
        "No": "No",
        "Maybe": "Maybe",
        
        # Feelings and States
        "Me Happy": "I am happy",
        "Me Sad": "I am sad",
        "Me Tired": "I am tired",
        "Me Hungry": "I am hungry",
        "Me Thirsty": "I am thirsty",
        "Me Sick": "I am sick",
        "Me Busy": "I am busy",
        "Me Good": "I am good",
        "Me Bad": "I am not feeling well",
        "Me Fine": "I am fine",
        "Me Okay": "I am okay",
        
        # Common Questions
        "How You": "How are you?",
        "What Time": "What time is it?",
        "What Day": "What day is it?",
        "What Date": "What is the date?",
        "Where Bathroom": "Where is the bathroom?",
        "Where Hospital": "Where is the hospital?",
        "Where Doctor": "Where is the doctor?",
        "Need Help": "I need help",
        "Help Me": "Please help me",
        "Call Help": "Please call for help",
        
        # Basic Needs
        "Want Water": "I want water",
        "Want Food": "I want food",
        "Want Rest": "I need to rest",
        "Want Sleep": "I need to sleep",
        "Need Medicine": "I need medicine",
        "Need Doctor": "I need a doctor",
        "Need Hospital": "I need to go to the hospital",
        
        # Family Related
        "My Family": "My family",
        "Me Mother": "My mother",
        "Me Father": "My father",
        "Me Sister": "My sister",
        "Me Brother": "My brother",
        "Me Parent": "My parents",
        "Me Child": "My child",
        "Me Baby": "My baby",
        
        # Work and Study
        "Me Work": "I am working",
        "Me Study": "I am studying",
        "Me Student": "I am a student",
        "Me Teacher": "I am a teacher",
        "Me Doctor": "I am a doctor",
        "Me Nurse": "I am a nurse",
        "Go Work": "I am going to work",
        "Go School": "I am going to school",
        "Go Home": "I am going home",
        
        # Time Related
        "Today": "Today",
        "Tomorrow": "Tomorrow",
        "Yesterday": "Yesterday",
        "Now": "Now",
        "Later": "Later",
        "Morning": "In the morning",
        "Afternoon": "In the afternoon",
        "Evening": "In the evening",
        "Night": "At night",
        
        # Emergency Related
        "Emergency": "This is an emergency",
        "Help Emergency": "I need emergency help",
        "Call Police": "Please call the police",
        "Call Ambulance": "Please call an ambulance",
        "Danger": "There is danger",
        "Fire": "There is a fire",
        
        # Common Actions
        "Me Go": "I am going",
        "Me Come": "I am coming",
        "Me Wait": "I am waiting",
        "Me Look": "I am looking",
        "Me Listen": "I am listening",
        "Me Speak": "I am speaking",
        "Me Write": "I am writing",
        "Me Read": "I am reading",
        
        # Social Interactions
        "Nice Meet You": "Nice to meet you",
        "Good Meet You": "Nice to meet you",
        "Meet Again": "Nice to meet you again",
        "Long Time": "Long time no see",
        "Miss You": "I miss you",
        "Love You": "I love you",
        "Care You": "I care about you",
        
        # Weather Related
        "Weather Hot": "The weather is hot",
        "Weather Cold": "The weather is cold",
        "Weather Rain": "It is raining",
        "Weather Snow": "It is snowing",
        "Weather Good": "The weather is good",
        "Weather Bad": "The weather is bad",
    }
    
    # Check for exact static phrases first
    if input_text in static_phrases:
        return static_phrases[input_text]
    
    # Question patterns
    question_patterns = {
        # Personal Questions
        "What You Name": "What is your name?",
        "What Name": "What is your name?",
        "Where You Live": "Where do you live?",
        "Where Live": "Where do you live?",
        "How Old You": "How old are you?",
        "What You Do": "What do you do?",
        "What Job": "What is your job?",
        "Where From": "Where are you from?",
        "Where You From": "Where are you from?",
        
        # Family Questions
        "Who You Mother": "Who is your mother?",
        "Who You Father": "Who is your father?",
        "Who You Parent": "Who are your parents?",
        "Who You Sister": "Who is your sister?",
        "Who You Brother": "Who is your brother?",
        "Have Family": "Do you have a family?",
        "Have Child": "Do you have children?",
        "Have Baby": "Do you have a baby?",
        
        # Time Questions
        "What Time Now": "What time is it now?",
        "What Day Today": "What day is it today?",
        "When You Come": "When are you coming?",
        "When You Go": "When are you going?",
        "When Start": "When does it start?",
        "When End": "When does it end?",
        
        # Location Questions
        "Where Go": "Where are you going?",
        "Where Now": "Where are you now?",
        "Where Here": "Where is this place?",
        "Where That": "Where is that?",
        "Where School": "Where is the school?",
        "Where Store": "Where is the store?",
        "Where Shop": "Where is the shop?",
        
        # Status Questions
        "How Feel": "How do you feel?",
        "Why Sad": "Why are you sad?",
        "Why Happy": "Why are you happy?",
        "Why Late": "Why are you late?",
        "Why Here": "Why are you here?",
        "What Wrong": "What is wrong?",
        "What Problem": "What is the problem?",
        
        # Action Questions
        "What Do": "What are you doing?",
        "What Want": "What do you want?",
        "What Need": "What do you need?",
        "What Like": "What do you like?",
        "What Eat": "What do you want to eat?",
        "What Drink": "What do you want to drink?",
        "What Help": "What help do you need?",
    }
    
    if input_text in question_patterns:
        return question_patterns[input_text]
    
    # Greeting patterns
    greeting_patterns = {
        r'^(Hi|Hello|Hey)\s+There$': lambda m: f"{m.group(1)} there!",
        r'^(Hi|Hello|Hey)\s+How\s+You$': lambda m: f"{m.group(1)}, how are you?",
        r'^(Hi|Hello|Hey)\s+([A-Z][a-zA-Z]*)$': lambda m: f"{m.group(1)} {m.group(2)}!",
        r'^Good\s+(Morning|Afternoon|Evening|Night)\s+([A-Z][a-zA-Z]*)$': 
            lambda m: f"Good {m.group(1).lower()} {m.group(2)}!",
    }
    
    # Check greeting patterns
    for pattern, replacement in greeting_patterns.items():
        match = re.match(pattern, input_text)
        if match:
            return replacement(match)
    
    # Main conversation patterns
    patterns = {
        # Personal Information
        r'^Me\s+Name\s+([A-Z][a-zA-Z]*)$': lambda m: f"My name is {m.group(1)}",
        r'^Me\s+([0-9]+)\s+Year$': lambda m: f"I am {m.group(1)} years old",
        r'^Me\s+From\s+([A-Z][a-zA-Z\s]*)$': lambda m: f"I am from {m.group(1)}",
        r'^Me\s+Live\s+([A-Z][a-zA-Z\s]*)$': lambda m: f"I live in {m.group(1)}",
        r'^Me\s+Like\s+([A-Z][a-zA-Z\s]*)$': lambda m: f"I like {m.group(1)}",
        r'^Me\s+Want\s+([A-Z][a-zA-Z\s]*)$': lambda m: f"I want {m.group(1)}",
        r'^Me\s+Need\s+([A-Z][a-zA-Z\s]*)$': lambda m: f"I need {m.group(1)}",
        
        # Possessive Patterns
        r'^My\s+([A-Z][a-zA-Z\s]*)$': lambda m: f"My {m.group(1).lower()}",
        r'^Your\s+([A-Z][a-zA-Z\s]*)$': lambda m: f"Your {m.group(1).lower()}",
        r'^Their\s+([A-Z][a-zA-Z\s]*)$': lambda m: f"Their {m.group(1).lower()}",
        
        # Action Patterns
        r'^Me\s+(Go|Come|Wait|Look|Listen|Speak|Write|Read)\s+([A-Z][a-zA-Z\s]*)$': 
            lambda m: f"I am {m.group(1).lower()}ing to {m.group(2)}",
        r'^You\s+(Go|Come|Wait|Look|Listen|Speak|Write|Read)\s+([A-Z][a-zA-Z\s]*)$': 
            lambda m: f"Are you {m.group(1).lower()}ing to {m.group(2)}?",
        
        # Time Patterns
        r'^(Today|Tomorrow|Yesterday|Now|Later)\s+([A-Z][a-zA-Z\s]*)$': 
            lambda m: f"{m.group(1)} {m.group(2).lower()}",
        r'^(Morning|Afternoon|Evening|Night)\s+([A-Z][a-zA-Z\s]*)$': 
            lambda m: f"In the {m.group(1).lower()} {m.group(2).lower()}",
        
        # Location Patterns
        r'^(Here|There|Where)\s+([A-Z][a-zA-Z\s]*)$': 
            lambda m: f"{m.group(1)} is {m.group(2).lower()}",
        r'^(Go|Come)\s+To\s+([A-Z][a-zA-Z\s]*)$': 
            lambda m: f"{m.group(1)} to {m.group(2).lower()}",
        
        # Feeling/State Patterns
        r'^(Feel|Look|Seem)\s+(Good|Bad|Happy|Sad|Tired|Sick)$': 
            lambda m: f"Feeling {m.group(2).lower()}",
        r'^You\s+(Good|Bad|Happy|Sad|Tired|Sick)$': 
            lambda m: f"Are you {m.group(1).lower()}?",
        
        # Question Patterns
        r'^(What|Where|When|Why|How|Who)\s+([A-Z][a-zA-Z\s]*)$': 
            lambda m: f"{m.group(1)} {m.group(2).lower()}?",
        r'^Can\s+([A-Z][a-zA-Z\s]*)$': 
            lambda m: f"Can {m.group(1).lower()}?",
        r'^Will\s+([A-Z][a-zA-Z\s]*)$': 
            lambda m: f"Will {m.group(1).lower()}?",
    }
    
    # Try each pattern and return the first match
    for pattern, replacement in patterns.items():
        match = re.match(pattern, input_text)
        if match:
            return replacement(match)
    
    # If no pattern matches, return the original text
    return input_text

# Interactive mode