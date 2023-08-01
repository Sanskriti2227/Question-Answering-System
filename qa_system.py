#Text-based Question Answering Task 
#import the required libraries 
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

#Data Preprocessig : For data preprocessing, we will write a function to clean and tokenize the context paragraphs and questions using the BERT tokenizer.
def preprocess_data(context, question):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    context_tokens = tokenizer.encode(context, add_special_tokens=True, max_length=512, truncation=True)
    question_tokens = tokenizer.encode(question, add_special_tokens=True, max_length=64)
    return context_tokens, question_tokens

#Fine-tune the QA Model: For fine-tuning, we will use a small dataset of context-question-answer triplets in SQuAD format. We'll use the Hugging Face Transformers library to fine-tune the pre-trained BERT model.
def fine_tune_QA_model(context, question, answer):
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    context_tokens, question_tokens = preprocess_data(context, question)
    start_pos, end_pos = tokenizer.encode(answer, add_special_tokens=False)

    input_ids = torch.tensor(context_tokens).unsqueeze(0)  # Batch size of 1
    start_pos = torch.tensor(start_pos).unsqueeze(0)  # Batch size of 1
    end_pos = torch.tensor(end_pos).unsqueeze(0)  # Batch size of 1

    outputs = model(input_ids=input_ids, start_positions=start_pos, end_positions=end_pos)
    loss = outputs.loss
    loss.backward()
    # Perform optimization here (you may use gradient descent or any other optimization algorithm)
    return model

#Command-Line User Interface: We will now create a basic command-line UI where users can input context paragraphs and questions, and the QA system can display the answers.
def answer_question(model, context, question):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    context_tokens, question_tokens = preprocess_data(context, question)

    input_ids = torch.tensor(context_tokens).unsqueeze(0)  # Batch size of 1
    start_scores, end_scores = model(input_ids=input_ids)

    start_pos = torch.argmax(start_scores)
    end_pos = torch.argmax(end_scores)

    answer_tokens = context_tokens[start_pos:end_pos + 1]
    answer = tokenizer.decode(answer_tokens)

    return answer

def main():
    context = input("Enter the context paragraph: ")
    question = input("Enter the question: ")

    model = fine_tune_QA_model(context, question, "Dummy answer")  # Replace "Dummy answer" with actual answer

    answer = answer_question(model, context, question)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
