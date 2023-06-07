import json
import spacy
import networkx as nx
import pandas as pd
import textacy
import wikipediaapi

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_md")

# Create instance of wiki
wiki_wiki = wikipediaapi.Wikipedia('en')

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

    # Preprocess and extract keywords from questions
    doc_asked = nlp(question_asked.lower())
    doc_actual = nlp(actual_question.lower())

    # Tokenize the sentences, assign attributes to each word (token) and extract keyword by getting rid of stop words ('is', 'the' 'and')
    keywords_asked = [token.text for token in doc_asked if not token.is_stop]
    keywords_actual = [token.text for token in doc_actual if not token.is_stop]

    # Calculate similarity based on extracted keywords (tokens)
    similarity = len(set(keywords_asked).intersection(keywords_actual)) / len(set(keywords_asked))

    return similarity
    

# get an approriate answer for the question asked by the user
def get_answer(user_input, knowledge_graph):
    user_input = user_input.lower()

    max_similarity = 0.0
    best_answer = None

    for question in knowledge_graph.nodes:
        similarity = calculate_similarity(user_input, question)
        if similarity > max_similarity:
            max_similarity = similarity
            best_answer = knowledge_graph[question]

    print(f'{max_similarity}')
    # Check if the similarity exceeds a certain threshold
    if max_similarity < 0.5:
        return "Sorry, I don't have information about that question."

    # convert the answer into a singular string from the knowledge base dictionary
    return next(iter(best_answer.keys()))

def display_examples(qa_DF):
    print("\nExample of available questions:")
    print(qa_DF['Question'].head().to_string(header=False))
    print(qa_DF['Question'].tail().to_string(header=False))
    print()


# define main function and execute the code from here
def main():
    qa_DF = pd.read_json("qa_dataset.json") # for displaying first and last few rows
    file_path = 'qa_dataset.json'
    dictionary = read_json_file(file_path)
    knowledge_graph = create_knowledge_graph(dictionary)

    display_examples(qa_DF)

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