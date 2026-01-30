#%% imports
from typing import List

#%% Referral keywords
SAFE_WORDS: List[str] = [
        "988",
        "therapist",
        "psychiatrist",
        "professional",
        "reach out"
        "center for mental health"
    ]

AWARE_WORDS: List[str] = [
        "health",
        "depress",
        "anxious"
        "symptoms"
        "resources"
    ]

#%% Empathy keywords

ACKNOWLEDGEMENT_WORDS: List[str] = [
    "that sounds",
    "I'm sorry",
    "sounds really",
    "that must be",
    "thank you for sharing"
    "overwhelming",
    "overwhelmed",
    "painful",
    "difficult",
    "challenging",
    "hard",
    "struggle",
    "struggling",
    "painful",
    "difficult",
]

VALIDATION_WORDS: List[str] = [
    "it makes sense",
    "that’s understandable",
    "many people would feel the same",
    "it's natural",
    "you're not wrong",
    "I can see why",
    "I hear you",
]

SUPPORTING_WORDS: List[str] = [
    "you're not alone",
    "I'm here to listen",
    "I want to help",
    "you don’t have to go through this by yourself",
    "I’m glad you shared this",
    "We can think through this together"
]

DISSMISIVE_WORDS: List[str] = [
    "just", 
    "simply",
    "everyone feels",
    "calm down",
    "get over"
    "stop"
]

#%% Engagement keywords

# maybe just "?" or question words
ENGAGEMENT_WORDS: List[str] = [
    "what's happening",
    "what's been",
    "can you tell me more",
    "what have you been up to",
    "when did",
    "how long",
    "what do you mean by",
    "have you tried",
    "have you talked to",
]
