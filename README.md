# NLP assignment README


## Student information

Name: Alexander Arkhipov<br>
Student number: 647833<br>
Python version: 3.9.13


## Steps to set-up the project

1. Make sure that you are using Python version 3.9 or above. For this assignment, version 3.9.13 was used.

2. Install `virtualenv` using the `pip` command

   ```bash
   pip install virtualenv
   ```

3. Create a virtual environment

   ```bash
   py -m venv venv OR python -m virtualenv venv
   ```

4. Enter the virtual environment

   ```bash
   # For Windows
   venv\Scripts\activate

   # For Linux/Mac
   source venv/bin/activate
   ```
5. Install the project dependencies from `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

## Running the application

1. Run the Python script/application.

   ```bash
   py chatbot.py OR python chatbot.py
   ```

## Quitting the application

1. Quit the Python script/application

   ```bash
   # When you are running the app use any of these three to quit by writing them into the console
   quit OR exit OR stop
   ```

## Description of the application

1. Purpose of the application
   This application is composed of a simple Chatbot that takes a question as the input and gives an answer as the output. 
   The Chatbot is trained based on a knowledge graph which consists of questions and answers that is retrieved from a JSON file.
   The data for the knowledge graph comes from random trivia questions. This data can be always added to the JSON file and the application will automatically update the dictionary after the application is relaunched.

2. How the application works
   The application makes use of the spacy model (en_core_web_md) which is in a nutshell serves as an English tokenizer, tagger, parser and NER (named entity recognition
   The application starts off by reading out the JSON file which contains the questions and answers and stores them into a dictionary where keys serve as the questions and answers as the values. It is important to note that for this application is only one answer per question.
   The application then consturcts a knowledge graph based on this dictionary where it loops through the items and adds question-answer pairs to the nodes and edges between the nodes to represent the relationship between them.
   The application then proceeds to ask the user for the input (question) for which it executes a method for retrieving the answer to the question.
   This function makes use of the knowledge graph by extracting the question nodes and calculating the similarity scores between the actual question and the ones asked by the user.
   Based on this, similarity scores will be calculated and it will take the highest similarity score and compares it to the threshold of 0.5 and if it is above that, it will return that as the answer, otherwise it will say that the question is not understood.
   The process of asking a question and receiving an answer will continue for as long as the user does not type "quit", "exit", or "stop".
   