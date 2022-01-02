# NER System

A web app for a simple named-entity recognition system

# Model

There are three models in this web app, two trained by me using Conditional Random Field, and one using spacy python package.

I have also trained another model using Distributed Random Forest with 92% accuracy, but the model can only be exported in Java, and Random Forest, but the model only reached 42% so lower than the CRF which has an accuracy of 85% when we disregard the O tag.

The two models are for Formal and Casual sentence. 

- Formal sentence means sentence with appropiate capitalization and correct English words and grammar.

- Casual sentence means sentence without correct capitalization and such.

# Web deployment

View the web app here: https://ner-simple-system.herokuapp.com/
