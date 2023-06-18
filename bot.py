import json
import spacy
import networkx as nx
import pandas as pd
import textacy
import wikipediaapi
import os
from fuzzywuzzy import fuzz
from spacy.lang.en.stop_words import STOP_WORDS
from networkx.readwrite import json_graph
from keytotext import pipeline
from spellchecker import SpellChecker

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_md")

# Load model for trnaslating keywords into sentences
nlp_ai = pipeline("k2t-base")

# Load coreferee
nlp.add_pipe('coreferee')

# Configure the model parameters
k2t_params = {"do_sample": True, "num_beams": 4, "no_repeat_ngram_size": 3, "early_stopping": True, "max_new_tokens":20}

# Create instance of wiki
wiki_wiki = wikipediaapi.Wikipedia('en')

# Method for extracting information from wiki based on the content name and saving it locally based on file location
def save_wiki_info(file_name, content_name):
    try:
        wiki_page = wiki_wiki.page(content_name)

        if not wiki_page.exists():
            print("Page does not exist")
            return

        content = wiki_page.text

        with open(file_name, "w", encoding='utf-8') as file:
            file.write(content)

        print("Page content has been saved locally")
    except Exception as e:
        print("An error occurred while saving the wiki page:", str(e))

# Method for saving content if there isnt any or reading out the content if it exists
def get_content(file_name):
    # Check if the data file exists, otherwise retrieve data from Wikipedia (adjust the content in the first parameter of the save_wiki_info() method)
    if not os.path.exists(file_name):
        save_wiki_info(file_name, "Gandhi")

    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            content = ' '.join(line.strip().lower() for line in file if line.strip())

    except FileNotFoundError:
        print('Cannot find local file, no knowledge base found.')

    return content
    

# Method for extracting the entity, object and relation triplets from the content provided and storing it into a dataframe
def construct_knowledge_graph(content):
    # Tokenize the content into sentences
    doc = nlp(content)
    lst_docs = [sent for sent in nlp(content).sents]

    # Preprocess the content
    content = preprocess_text(content)

    # Print the total number of sentences found
    print("Total sentences found in dataset:", len(lst_docs))

    data = []

    def coref(parts):
        # Handle coreference resolution for each part
        for part in parts:
            subj_ref = doc._.coref_chains.resolve(part)
            if subj_ref:
                for x in subj_ref:
                    yield x
            else:
                yield part

    for i, sentence in enumerate(lst_docs):
        for sent in textacy.extract.subject_verb_object_triples(sentence):
            subj = " ".join(map(str, coref(sent.subject)))
            obj = " ".join(map(str, coref(sent.object)))
            relation = " ".join(map(str, sent.verb))

            # Create a data entry for each subject-verb-object triple
            data.append({
                "id": i,
                "text": sentence.text,
                "entity": subj,
                "object": obj,
                "relation": relation
            })

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Filter to avoid duplicates based on the most frequent entity
    f = df["entity"].value_counts().head().index[0]
    tmp = df[(df["entity"] == f) | (df["object"] == f)]

    # Construct the knowledge graph using the filtered DataFrame
    G = nx.from_pandas_edgelist(tmp, source="entity", target="object", edge_attr="relation", create_using=nx.DiGraph())
    return G


# Serialize the knowledge graph to a file in GML format
def serialize_knowledge_graph(knowledge_graph, file_name):
    nx.write_gml(knowledge_graph, file_name)
    print("Knowledge graph has been serialized and saved.")


# Method for loading in graph, if one exists, otherwise, make a new one, save it and read from it
def load_knowledge_graph(file_name_kg, file_name_data):

    # If the knowledge graph or data file does not exist, create a new knowledge graph, serialize it, and read it out
    if not os.path.exists(file_name_kg) or not os.path.exists(file_name_data):
        content = get_content("data.txt")
        knowledge_graph = construct_knowledge_graph(content)
        print("Knowledge graph is being serialized...")
        serialize_knowledge_graph(knowledge_graph, file_name_kg)
    
    # If the knowledge graph exists, read it from the file
    knowledge_graph = nx.read_gml(file_name_kg)

    return knowledge_graph


# preprocess question of user
def preprocess_text(input_text):
    # Convert input text to lowercase
    input_text = input_text.lower()

    # Remove question mark
    input_text = input_text.rstrip("?")

    # Remove stop words
    doc = nlp(input_text)
    tokens = [token.text for token in doc if token.text.lower() not in STOP_WORDS]
    preprocessed_text = " ".join(tokens)

    # Perform spelling correction
    spell_checker = SpellChecker(language='en')
    corrected_text = spell_checker.correction(preprocessed_text)

    # If text was corrected use it, otherwise use the preprocessed text
    handed_text = corrected_text if corrected_text else preprocessed_text

    return handed_text


def generate_sentence(source, relation, dest, config):
    return nlp_ai([source, relation.get('relation'), dest], **config)


def get_answer(user_input, knowledge_graph):
    # Preprocess the user input
    user_input = preprocess_text(user_input)

    # Calculate similarities between the user input and each node in the knowledge graph
    similarities = {node: nlp(user_input).similarity(nlp(node)) for node in knowledge_graph.nodes}

    # Find the node with the highest similarity score
    top_answer = max(similarities, key=similarities.get)

    relationship = None
    response = None
    found_response = False

    # Iterate over the outgoing edges of the top answer node
    for source, dest, relation in knowledge_graph.out_edges(top_answer, data=True):
        # Check if the source node matches the top answer or the relation matches the previous relation
        if source == top_answer or relation.get('relation') == relationship:
            # Generate the response sentence
            response = generate_sentence(source, relation, dest, k2t_params)
            found_response = True
            break

    # If no response is found, set a default response
    if not found_response:
        response = "I'm sorry, but I couldn't find an answer to your question."

    return response


# Define main function and execute the code from here
def main():
    # Load the knowledge graph and data
    knowledge_graph = load_knowledge_graph("knowledge_graph.GML", "data.txt")

    while True:
        # Get user input
        user_input = input("Your question: ")

        # Check if the user wants to exit
        if user_input in ['exit', 'quit', 'stop']:
            break

        # Get the answer from the knowledge graph
        answer = get_answer(user_input, knowledge_graph)

        # Print the chatbot's response
        print(f"Chatbot's response: {answer}")

    print("Thank you for using the chatbot. Goodbye!")


# Call the main function
if __name__ == '__main__':
    main()