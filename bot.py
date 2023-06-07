import json
import spacy
import networkx as nx
import pandas as pd
import textacy
import wikipediaapi
import os
import matplotlib.pyplot as plt
import pickle
from fuzzywuzzy import fuzz
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_md")

# Create instance of wiki
wiki_wiki = wikipediaapi.Wikipedia('en')

# Set up stopwords and lemmatizer
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Method for extracting information from wiki based on the content name and saving it locally based on file location
def save_wiki_info(content_name, file_location):
    wiki_page = wiki_wiki.page(content_name)

    if not wiki_page.exists():
        print("Page does not exist")
        return

    content = wiki_page.text

    with open(file_location, "w", encoding='utf-8') as file:
        file.write(content)

    print("Page content has been saved locally")

# Method for saving content if there isnt any or reading out the content if it exists
def get_content(file_name):
    # if data file does not exist, get data from wikiapi (keep in mind, you have to adjust the content of it in the first parameter of save_wiki_info() method)
    if not os.path.exists(file_name):
        save_wiki_info("Apple", file_name)

    with open(file_name, "r", encoding='utf-8') as file:
        content = file.read()
    return content

# Method for extracting the entity, object and relation triplets from the content provided and storing it into a dataframe
def extract_relations_and_entities(content):

    lst_docs = [sent for sent in nlp(content).sents]

    data = []

    for i, sentence in enumerate(lst_docs):
        for sent in textacy.extract.subject_verb_object_triples(sentence):
            subj = " ".join(map(str, sent.subject))
            obj = " ".join(map(str, sent.object))
            relation = " ".join(map(str, sent.verb))

            data.append({
                "id": i,
                "text": sentence.text,
                "entity": subj,
                "object": obj,
                "relation": relation
            })

    return pd.DataFrame(data)

# Method for constructing a knowledge graph based on the extracted triplets
def construct_knowledge_graph(df):
    G = nx.from_pandas_edgelist(df, source="entity", target="object", edge_attr="relation", create_using=nx.DiGraph())
    return G

# Method for saving the knowledge graph so it can be reused later instea of constantly initializing it
def serialize_knowledge_graph(knowledge_graph, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(knowledge_graph, file)
    print("Knowledge graph has been serialized and saved.")

# Method for loading in graph, if one exists, otherwise, make a new one, save it and read from it
def load_knowledge_graph(file_name):

    #if knowledge graph does not exist yet, make it and serialize it, then read it out
    if not os.path.exists(file_name):
        content = get_content("data.txt")
        df = extract_relations_and_entities(content)
        knowledge_graph = construct_knowledge_graph(df)
        serialize_knowledge_graph(knowledge_graph, file_name)
    
    # if knowledge graph exist, just read it out
    with open(file_name, "rb") as file:
        knowledge_graph = pickle.load(file)
    return knowledge_graph

# define main function and execute the code from here
def main():

    knowledge_graph = load_knowledge_graph("knowledge_graph.pickle")
    print(knowledge_graph)
    
    # while True:

    #     user_input = input("Your question: ")

    #     if user_input in ['exit', 'quit', 'stop']:
    #         break

    #     preprocessed_input = preprocess_input(user_input)

    #     best_match, best_score = apply_fuzzy_matching(preprocessed_input, knowledge_graph)
        
    #     chatbot_response = get_answer(best_match, best_score)

    #     print(f"Chatbot's response: {chatbot_response}")

    # print("Thank you for using the chat bot, hope to see you again soon!")

    
   

# Call the main function
if __name__ == '__main__':
    main()