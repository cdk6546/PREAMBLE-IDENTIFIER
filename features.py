import pandas as pd

hungarian_list = ['b', 'ch', 'c', 'dw', 'f', 'n', 'i', 'fp', 'db', 'p', 'rg', 'sz', 'u16', 'u32', 'st', 'fn']


def create_features(data):
    data = word_total(data)
    # data = file_extension(data)
    data = word_length(data)
    data = first_letter_capitalized(data)
    data = hungarian_notation(data)
    return data


def word_total(data):
    identifier_data = data['IDENTIFIER']
    word_total = pd.DataFrame(columns=['WORD_TOTAL'])
    for element in identifier_data:
        identifier_length = len(element.split())
        word_total.loc[len(word_total), 'WORD_TOTAL'] = identifier_length
    data = pd.concat([data, word_total], axis=1)
    return data


# this one seems like it most likely does not change anything in the accuracy
# def file_extension(data):
#     file_data = data['SYSTEM']
#     file_ext = pd.DataFrame(columns=['FILE_EXT'])
#     for element in file_data:
#         string_split = element.split(".")
#         if len(string_split) > 1:
#             current_ext = string_split[1]
#             file_ext.loc[len(file_ext), 'FILE_EXT'] = current_ext
#         else:
#             file_ext.loc[len(file_ext), 'FILE_EXT'] = "NONE"
#
#     data = pd.concat([data, file_ext], axis=1)
#     return data


def word_length(data):
    identifier_data = data['WORD']
    word_length = pd.DataFrame(columns=['WORD_LENGTH'])
    for element in identifier_data:
        length = len(element)
        word_length.loc[len(word_length), 'WORD_LENGTH'] = length
    data = pd.concat([data, word_length], axis=1)
    return data


def first_letter_capitalized(data):
    identifier_data = data['IDENTIFIER']
    first_word_cap = pd.DataFrame(columns=['FIRST_WORD_CAP'])

    for element in identifier_data:
        words = element.split(' ')
        first_word = words[0]
        first_letter = first_word[0]

        if ord(first_letter) >= 97 and ord(first_letter) <= 122:
            first_word_cap.loc[len(first_word_cap), 'FIRST_WORD_CAP'] = False
        elif ord(first_letter) >= 65 and ord(first_letter) <= 90:
            first_word_cap.loc[len(first_word_cap), 'FIRST_WORD_CAP'] = True
    data = pd.concat([data, first_word_cap], axis=1)
    return data


def hungarian_notation(data):
    identifier_data = data['IDENTIFIER']
    hungarian = pd.DataFrame(columns=['HUNGARIAN'])
    true_count = 0

    for element in identifier_data:
        words = element.split(' ')
        first_word = words[0]
        if first_word in hungarian_list:
            hungarian.loc[len(hungarian), 'HUNGARIAN'] = True
            true_count += 1
        else:
            hungarian.loc[len(hungarian), 'HUNGARIAN'] = False
    data = pd.concat([data, hungarian], axis=1)

    # to make sure the feature is actually working
    print(f"true count: {true_count}")
    return data
