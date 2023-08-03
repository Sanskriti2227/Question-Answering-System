from transformers import BertTokenizer, BertForQuestionAnswering
import torch

def initialize_qa_model():
    model_path = "fine_tuned_bert_model"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained(model_path)

    return tokenizer, model

def get_answer(context, question, tokenizer, model):
    inputs = tokenizer(context, question, return_tensors='pt')
    start_scores, end_scores = model(**inputs).start_logits, model(**inputs).end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index + 1]))

    return answer

if __name__ == "__main__":
    context = "Who did the first work generally recognized as AI?"
    question = "Warren McCulloch and Walter Pitts (1943)"

    tokenizer, model = initialize_qa_model()
    answer = get_answer(context, question, tokenizer, model)

    print("Answer:", answer)
