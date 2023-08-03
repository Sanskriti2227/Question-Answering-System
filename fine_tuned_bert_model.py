import pandas as pd
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
import torch

def fine_tune_bert(pylance):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    # Read the preprocessed dataset
    df = pd.read_csv("C:\Users\HP\OneDrive\Desktop\QuestionAnsweringSystem")

    # Tokenize and prepare inputs for the model
    inputs = tokenizer(list(df['context']), list(df['question']), return_tensors='pt', padding=True, truncation=True)

    # Prepare target answer positions for the model
    start_positions = torch.tensor(df['start_position'].tolist())
    end_positions = torch.tensor(df['end_position'].tolist())

    # Fine-tune the model
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model.train()
    for epoch in range(5):  # 5 epochs as an example; you can adjust this
        optimizer.zero_grad()
        outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_bert_model")

if __name__ == "__main__":
    processed_dataset_file = "processed_qa_dataset.csv"
    fine_tune_bert(processed_dataset_file)
