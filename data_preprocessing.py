import pandas as pd
from transformers import BertTokenizer

def preprocess_data(filepath):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Read the dataset from CSV
    df = pd.read_csv("C:\Users\HP\OneDrive\Desktop\QuestionAnsweringSystem\qa_dataset.csv")

    # Clean and tokenize context and questions
    df['context_tokens'] = df['context'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True))
    df['question_tokens'] = df['question'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

    return df

if __name__ == "__main__":
    dataset_file = "qa_dataset.csv"  
    processed_data = preprocess_data(dataset_file)
    processed_data.to_csv("processed_qa_dataset.csv", index=False)
