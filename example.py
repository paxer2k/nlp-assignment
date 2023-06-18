import spacy
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
import textacy
import wikipediaapi
from spellchecker import SpellChecker
from spacy.lang.en.stop_words import STOP_WORDS
import json
from fuzzywuzzy import fuzz
from keytotext import pipeline

 

print("Preparing gandhibot...")

 

# get and clean data
def get_data():
    # get text data from wikipedia about bicycles
    try:
        print('Downloading new data...')
        wiki_wiki = wikipediaapi.Wikipedia('en')
        page_py = wiki_wiki.page('Mahatma_Gandhi')
        file = open(r"./data/data.txt", "w+")
        file.write(page_py.text)
        file.close()
    except ConnectionError:
        print('Cant get new data, using local data.')

    # Read data from json file
    try:
        with open('./data/data.txt', 'r') as f:
            raw = f.readlines()

        # filter newlines, empty strings, lowercase and combine into string 
        txt = list(map(str.strip, raw))
        txt = list(map(lambda x: x.lower(), txt))
        txt = list(filter(None, txt))
        txt = ' '.join(map(str, txt))
        # print(json.dumps(txt, indent=4))
    except FileNotFoundError:
        print('Cant find local, no knowledge base found.')

 

    return txt

# knowledge graph
def create_knowledge_graph(txt):
    # from text to a list of sentences
    lst_docs = [sent for sent in nlp(txt).sents]

    doc = nlp(txt)


 

    txt = preprocess_text(txt)

    print("Total sentences found in dataset:", len(lst_docs))

 

    # extract entities and relations
    dic = {"id": [], "text":[], "entity":[], "relation":[], "object":[]}

 

    def coref(parts):
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
            obj  = " ".join(map(str, coref(sent.object)))
            relation = " ".join(map(str, sent.verb))

            dic["id"].append(i)
            dic["text"].append(sentence.text)
            dic["entity"].append(subj)
            dic["object"].append(obj)
            dic["relation"].append(relation)

 

    ## create dataframe
    dtf = pd.DataFrame(dic)

 

    ## filter
    f = dtf["entity"].value_counts().head().index[0]
    tmp = dtf[(dtf["entity"]==f) | (dtf["object"]==f)]

 

    
    ## create small graph
    graph = nx.from_pandas_edgelist(
        tmp, 
        source="entity", 
        target="object", 
        edge_attr="relation", 
        create_using=nx.DiGraph()
        )

 

    ## plot
    plt.figure(figsize=(15,10))
    pos = nx.spring_layout(graph, k=1)

    node_color = ["red" if node==f else "skyblue" for node in graph.nodes]
    edge_color = ["red" if edge[0]==f else "black" for edge in graph.edges]

    nx.draw(
        graph, 
        pos=pos, 
        with_labels=True, 
        node_color=node_color, 
        edge_color=edge_color,
        node_size=2000, 
        node_shape="o"
        )

 

    nx.draw_networkx_edge_labels(
        graph, 
        pos=pos, 
        label_pos=0.5, 
        edge_labels=nx.get_edge_attributes(graph,'relation'),
        font_size=12, 
        font_color='black', 
        alpha=0.6)

    # warning about missing font in ./.cache/matplotlib
    plt.rcParams['axes.unicode_minus']=False 
    plt.rc('axes', unicode_minus=False)
    plt.savefig('knowledge_graph.jpg', bbox_inches='tight')

 

    synonym(graph)
    nx.write_gml(graph, "knowledge_graph.GML")
    return graph

 

# adding synonym to graph
def synonym(graph):
    with open('./data/synonym.json', 'r') as j:
        synonyms = json.load(j)

    for original, synonym_list in synonyms.items():
        for synonym in synonym_list:
            if graph.nodes.__contains__(synonym):
                graph.add_edge(original, synonym, relation="Synonym")

 

    return graph

 

# preprocess question of user
def preprocess_text(input_text):
    # Convert input text to lowercase
    input_text = input_text.lower()

 

    # Remove question mark
    input_text = input_text.replace("?", "")

    # Remove stop words
    doc = nlp(input_text)
    tokens = [token.text for token in doc if token.text.lower() not in STOP_WORDS]
    preprocessed_text = " ".join(tokens)

 

    # Perform spelling correction
    spell_checker = SpellChecker(language='en')
    query = spell_checker.correction(preprocessed_text)
    if query is None:
        query = preprocessed_text

 

    return query

 

def create_centence(source, relation, dest, config): 
    return nlp_ai([source, relation.get('relation'), dest], **config)

 

# define function to get a response from the chatbot
def get_response(input_text, graph):
    # convert input text to all lowercase to avoid case sensitivity
    input_text = input_text.lower()

 

    if input_text == ['exit', 'stop', 'bye']:
        return None

 

    input = preprocess_text(input_text)

 

    highest_score = 0
    nodes = {}
    for node in graph.nodes:
        similarity = nlp(input).similarity(nlp(node))
        if similarity > highest_score:
            highest_score = similarity
            nodes[node] = [highest_score, graph[node]]

    top_answer = max(nodes, key=nodes.get)

    for source, dest, relation in graph.out_edges(top_answer, data=True):
        if source == top_answer:
            relationship = relation.get('relation')

    response = None
    config = {"do_sample": True, "num_beams": 4, "no_repeat_ngram_size": 3, "early_stopping": True, "max_new_tokens":20}
    for source, dest, relation in graph.out_edges(top_answer, data=True):
        if source == top_answer:
            response = create_centence(source, relation, dest, config)
            break
        elif(relation.get('relation') == relationship):
            response = create_centence(source, relation, dest, config)
            break
    if not response:
        response = "I'm sorry but I couldn't find an answer to your question."
    return response


 

# load spacy with english and model
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('coreferee')
nlp_ai = pipeline("k2t-base")

 

user_input = input('Check for new data?(yes/no) ')
if((user_input.lower().find('y')) == 0):
    txt = get_data()
    graph = create_knowledge_graph(txt)
else:
    print('Using local data.')
    graph = nx.read_gml("knowledge_graph.GML")

 

print("Hi, I am Gandhibot, a chatbot specialized about Mahatma Gandhi. Ask your question about Mahatma Gandhi.")

 

# loop to continuously prompt user for input and return chatbot response
while True:
    user_input = input("You: ")

    chatbot_response = get_response(user_input, graph)

    if not chatbot_response:
        break
    print("Gandhibot: " + chatbot_response)

 

print("Thanks for chatting with gandhibot.")