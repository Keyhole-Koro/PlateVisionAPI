import re

def is_valid_hiragana(hiragana):
    return bool(re.fullmatch(r'[\u3040-\u309F]', hiragana))

def is_valid_number(number):
    """Check if the number is valid, allowing '-' and '・' in between."""
    if not isinstance(number, str):
        return False
    
    # Ensure the number contains at least one digit
    if not re.search(r'\d', number):
        return False

    # Allow only digits, '-', and '・' in the original input
    if not re.fullmatch(r'[\d・\-]+', number):
        return False

    # Remove '-' and '・' for final validation
    cleaned_number = re.sub(r'[-・]', '', number)

    # Check if the cleaned number is 1-4 digits long and starts with 1-9
    return bool(re.fullmatch(r'[1-9]\d{0,3}', cleaned_number))

def is_valid_classification(classification):
    # Check if classification has exactly 3 digits
    return re.fullmatch(r'\d{3}', classification) is not None

def is_valid_result(hiragana, number, classification):
    return is_valid_hiragana(hiragana) and is_valid_number(number) and is_valid_classification(classification)