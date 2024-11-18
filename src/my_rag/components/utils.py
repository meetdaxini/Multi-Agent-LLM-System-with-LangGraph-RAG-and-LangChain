import re


def alphanumeric_string(input_string):
    return re.sub(r"[^a-zA-Z0-9]", "", input_string)
