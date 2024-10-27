import json
import pandas as pd


def json_file_to_csv(json_file_path, csv_file_path):
    """
    Reads JSON data from a file, converts it to a table format, and saves it as a CSV file.

    Args:
    json_file_path (str): The file path of the JSON file to be read.
    csv_file_path (str): The file path where the CSV file will be saved.
    """
    # Read JSON data from the file
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    data = []

    for question in json_data['questions']:
        body = question['body']
        documents = "; ".join(question['documents'])  # Joining document links
        ideal_answer = "; ".join(question['ideal_answer'])  # Joining ideal answers
        # concepts = "; ".join(question['concepts'])  # Joining concepts
        question_type = question['type']
        question_id = question['id']

        for snippet in question['snippets']:
            snippet_data = {
                'body': body,
                'documents': documents,
                'ideal_answer': ideal_answer,
                #'concepts': concepts,
                'type': question_type,
                'id': question_id,
                'snippet_text': snippet['text'],
                'offsetInBeginSection': snippet['offsetInBeginSection'],
                'offsetInEndSection': snippet['offsetInEndSection'],
                'beginSection': snippet['beginSection'],
                'endSection': snippet['endSection'],
                'document': snippet['document']
            }
            data.append(snippet_data)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(csv_file_path, index=False)

    return df


#%%

# Example usage
json_file_path = '/Capstone Project Fall 2024 /Data/BioASQ-training11b/training11b.json'
csv_file_path = '/Capstone Project Fall 2024 /output_JSON.csv'

df = json_file_to_csv(json_file_path, csv_file_path)
print(f"CSV saved at {csv_file_path}")


#%%

data = pd.read_csv('Capstone Project Fall 2024 /output_JSON.csv')
print(data.shape)