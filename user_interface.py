from question_answering import initialize_qa_model, get_answer

def main():
    print("Text-based Question Answering System")
    print("Enter 'exit' to quit.")

    tokenizer, model = initialize_qa_model()

    while True:
        context = input("Enter the context paragraph: ")
        if context.lower() == 'exit':
            break

        question = input("Enter the question: ")
        if question.lower() == 'exit':
            break

        answer = get_answer(context, question, tokenizer, model)
        print("Answer:", answer)

if __name__ == "__main__":
    main()
