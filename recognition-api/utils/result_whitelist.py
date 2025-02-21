import re

def is_valid_hiragana(hiragana):
    return bool(re.fullmatch(r'[\u3040-\u309F]', hiragana))

def is_valid_number(number):
    # Remove '-' and '・' from the number
    cleaned_number = number.replace('-', '').replace('・', '')
    # Check if the cleaned number has 1-4 digits
    return re.fullmatch(r'\d{1,4}', cleaned_number) is not None

def is_valid_classification(classification):
    # Check if classification has exactly 3 digits
    return re.fullmatch(r'\d{3}', classification) is not None

def is_valid_result(hiragana, number, classification):
    return is_valid_hiragana(hiragana) and is_valid_number(number) and is_valid_classification(classification)