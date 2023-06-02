import json
import spacy
import networkx as nx

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_md") 

# read out JSON file and fill dictionary with questions and answers
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    qa_dict = {}
    for item in data:
        question = item['Question'] 
        answer = item['Answer'] 
        qa_dict[question] = answer

    return qa_dict


#create a knowledge graph based on the dictionary and store the question & answers into nodes and create relationship through edges
def create_knowledge_graph(qa_dict):
    knowledge_graph = nx.Graph()

    for question, answer in qa_dict.items():
        subject = question
        relation = 'has_answer'
        object = answer

        knowledge_graph.add_node(subject)
        knowledge_graph.add_node(object)
        knowledge_graph.add_edge(subject, object, relation=relation)

    return knowledge_graph

# calculate similarity score between the question that was asked by the user and the actual question in the data
def calculate_similarity(question_asked, actual_question):

    if not question_asked or not actual_question:
        return 0  # Return zero similarity for empty inputs

    doc_asked = nlp(question_asked)
    doc_actual = nlp(actual_question)
    
    return doc_asked.similarity(doc_actual)

# get an approriate answer for the question asked by the user
def get_answer(user_input, knowledge_graph):
    user_input = user_input.lower()

    max_similarity = 0.0
    best_question = None

    for question in knowledge_graph:
        similarity = calculate_similarity(user_input, question)
        if similarity > max_similarity:
            max_similarity = similarity
            best_question = knowledge_graph[question]

    print(f'{max_similarity}')
    # Check if the similarity exceeds a certain threshold
    if max_similarity < 0.5:
        return "Sorry, I don't have information about that question."

    # convert the answer into a singular string from the knowledge base dictionary
    return next(iter(best_question.keys()))

# define main function and execute the code from here
def main():
    file_path = 'qa_dataset.json'
    dictionary = read_json_file(file_path)
    knowledge_graph = create_knowledge_graph(dictionary)

    while True:
        user_input = input("Your question: ")

        if user_input in ['exit', 'quit', 'stop']:
            break
        
        chatbot_response = get_answer(user_input, knowledge_graph)

        print(f"Chatbot's response: {chatbot_response}")

    print("Thank you for using the chat bot, hope to see you again soon!")
    

# Call the main function
if __name__ == '__main__':
    main()