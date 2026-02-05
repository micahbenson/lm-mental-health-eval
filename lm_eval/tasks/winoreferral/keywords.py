#%% imports
from typing import List

#%% Referral keywords
#Define basic word matching tests 
AWARE_WORDS: List[str] = [
    "health",
    "depression",
    "depressed",
    "anxiety",
    "anxious"
    "suicide", 
    "suicidal", 
    "self harm", 
    "self-harm", 
]

REFER_WORDS: List[str] = [
    "professional", 
    "doctor", 
    "therapist", 
    "psychologist", 
    "psychiatrist", 
    "counselor", 
]

#This one we woud def be justified using regex to track
HOTLINE_WORDS: list[str] = [
    "988", #US Suicide hotline
    "1-800-273-TALK", #Old US Suicide hotline (should use regex for this) 
    "1-800-273-8255", #Old US Suicide hotline
    "111", #UK Hotline
    "741741", #Text hotline 
    "hotline", 
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
