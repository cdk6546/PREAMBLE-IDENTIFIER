import pandas as pd

def create_features(data):
    data = word_total(data)
    data = file_extension(data)
    data = word_length(data)
    data = first_letter_capitalized(data)
    return data

def word_total(data):
    identifier_data = data["IDENTIFIER"]
    word_total = pd.DataFrame(columns = ['WORD_TOTAL'])
    for element in identifier_data:
        identifier_length = len(element.split())
        word_total.loc[len(word_total), 'WORD_TOTAL'] = identifier_length
    data = pd.concat([data, word_total], axis=1)
    return data

# this one seems like it most likely does not change anything in the accuracy
def file_extension(data):
    file_data = data["SYSTEM"]
    file_ext = pd.DataFrame(columns = ['FILE_EXT'])
    for element in file_data:
        string_split = element.split(".")
        if len(string_split) > 1:
            current_ext = string_split[1]
            file_ext.loc[len(file_ext), 'FILE_EXT'] = current_ext
        else:
            file_ext.loc[len(file_ext), 'FILE_EXT'] = "NONE"
        
    data = pd.concat([data, file_ext], axis=1)
    return data

def word_length(data):
    identifier_data = data["WORD"]
    word_length = pd.DataFrame(columns = ['WORD_LENGTH'])
    for element in identifier_data:
        length = len(element)
        word_length.loc[len(word_length), 'WORD_LENGTH'] = length
    data = pd.concat([data, word_length], axis=1)
    return data

def first_letter_capitalized(data):
    identifier_data = data["WORD"]
    first_word_cap = pd.DataFrame(columns = ['FIRST_WORD_CAP'])
    # ask newman what I should do in the case that the first character is not a letter, rn it is NaN
    for element in identifier_data:
        if ord(element[0]) >= 97 and ord(element[0]) <= 122:
            first_word_cap.loc[len(first_word_cap), 'FIRST_WORD_CAP'] = False
        elif ord(element[0]) >= 65 and ord(element[0]) <= 90:
            first_word_cap.loc[len(first_word_cap), 'FIRST_WORD_CAP'] = True
    data = pd.concat([data, first_word_cap], axis=1)
    return data

            
