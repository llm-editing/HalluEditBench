import re
import os
import json
import time
import pickle
import requests
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from bs4 import BeautifulSoup
from urllib.error import HTTPError
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor

import spacy
from flair.models import SequenceTagger
from flair.data import Sentence


sequence_tagger = SequenceTagger.load('ner')
spacy_en_core_web = spacy.load("en_core_web_lg")
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")


def identifier_conversion(entity, property=False):
    if not property:  # 'city'
        query = f"""
            SELECT ?identifier WHERE {{
                ?identifier rdfs:label "{entity}"@en.
            }}
            """
    else:  # 'instance of'
        query = f""" 
            SELECT ?identifier WHERE {{
                ?property rdf:type wikibase:Property .
                ?identifier rdfs:label "{entity}"@en. 
            }}
            """
    property_pattern = r'^P\d+'
    node_pattern = r'^Q\d+'
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if "results" in results and "bindings" in results["results"]:
        if not property:
            for result in results["results"]["bindings"]:
                identifier = result["identifier"]["value"].split("/")[-1]
                if re.match(node_pattern, identifier):
                    return identifier
        else:
            for result in results["results"]["bindings"]:
                identifier = result["identifier"]["value"].split("/")[-1]
                if re.match(property_pattern, identifier):
                    return identifier
    return None


def convert_topic_to_symbol(topic_dict):
    relation_object_pairs = []
    for key, value in topic_dict.items():
        key = identifier_conversion(key, True)
        value = identifier_conversion(value)
        if key and value:
            relation_object_pairs.append([key, value])
        else:
            raise Exception(f"'{key}: {value}' cannot be converted to identifier!")
    return relation_object_pairs


def process_result(result):
    subject_label = result["subjectLabel"]["value"]
    relation_label = result["relation"]["value"]
    try:
        reference_response = requests.get(relation_label)
        reference_soup = BeautifulSoup(reference_response.content, 'html.parser')
        relation_label = reference_soup.find("span", class_="wikibase-title-label")
    except requests.exceptions.RequestException as e:
        # Handle the connection error
        print(f"Connection error occurred for relation '{relation_label}': {e}")
        return None
    object_label = result["objectLabel"]["value"]

    return {
        "subjectLabel": subject_label,
        "relation": relation_label.text,
        "objectLabel": object_label
    }
    

def get_topic_size(topics):
    for topic in topics:
        if topic:
            topic = json.loads(topic)
            query_part1 = "SELECT ?subjectLabel ?relation ?objectLabel WHERE {"
            query_part2 = ""
            relation_object_pairs = convert_topic_to_symbol(topic)
            for pair in relation_object_pairs:
                query_part2 += f"\n?subject wdt:{pair[0]} wd:{pair[1]} ."
            query_part3 = """
                ?subject  ?relation  ?object.
                ?subject wikibase:identifiers ?subject_identifierCount.
                ?object wikibase:identifiers ?object_identifierCount.
                """
            query_part5 = """ 
                FILTER (?subject_identifierCount >= 8 && ?object_identifierCount >= 5) .  
                SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
            }
            LIMIT 8000
            """
            query = query_part1 + query_part2 + query_part3 + query_part5
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            count = len(results['results']['bindings'])
            print(f"Topic {topic} size: {count}")
            return count


def generate_question(subject, relation, object, topic, query_subject=False):
    object_type1 = None
    object_type2 = None
    object_type = None
    discard_flag = False
    convert_dict1 = {
        "PER": "PERSON",
        "LOC": "GPE"
    }

    ####### method 1
    sentence = Sentence(object)
    # Predict entities
    sequence_tagger.predict(sentence)
    # Access entity annotations
    entities = sentence.get_spans('ner')
    # Print the recognized entities
    if entities:
        object_type1 = entities[0].tag
        if object_type1 == "PER" or object_type1 == "LOC":
            object_type1 = convert_dict1[object_type1]
        else:
            object_type1 = None

    ####### method 2
    object_doc = spacy_en_core_web(object)
    if object_doc.ents:
        object_type2 = object_doc.ents[0].label_

    if object_type1:        
        if object_type1 == object_type2:
            object_type = object_type1
        else:
            discard_flag = True
    else:
        if object_type2 != "GPE" and object_type2 != "PERSON":
            object_type = object_type2
        else:
            discard_flag = True
            
    if discard_flag:
        return None

    subject_doc = spacy_en_core_web(relation)

    if subject_doc[-1].tag_ == "IN" and subject_doc[0].tag_ not in ["VBN", "VBD", "VB", "VBZ"]:
        return None
        
    question_answer_pair = {}
    question_answer_pair["subject"] = subject
    question_answer_pair["relation"] = relation
    question_answer_pair["object"] = object

    relation_set = set()
    for token in subject_doc:
        relation_set.add(token.tag_)

    object_to_interrogative = {
        "PERSON": "Who",
        "DATE": "When",
    }

    default_interrogative = "What"  # Default value      
    interrogative = object_to_interrogative.get(object_type, default_interrogative)
    if query_subject:
        tmp = subject
        subject = object
        object = tmp

    if subject_doc[0].tag_ == "VBN" and subject_doc[-1].tag_ == "IN" and all(token.tag_ not in ["NN", "NNP", "NNPS", "NNS"] for token in subject_doc[0:]):
        if not query_subject:
            question_answer_pair["question"] = interrogative + " was " + subject + " " + relation + "?"
            question_answer_pair["label"] = object
        else:
            if object_type != "PERSON":
                first_pair = next(iter(topic.items()))
                if first_pair[1] != "revolution":
                    interrogative = "Which " + first_pair[1]
                else:
                    interrogative = "Which revolution or war"
            
            question_answer_pair["question"] = interrogative + " was " + relation + " " + object + "?"
            question_answer_pair["label"] = subject

    elif subject_doc[0].tag_ == "JJ" and subject_doc[-1].tag_ == "IN" and all(token.tag_ not in ["NN", "NNP", "NNPS", "NNS"] for token in subject_doc[0:]):
        if not query_subject:
            question_answer_pair["question"] = interrogative + " is " + subject + " "+ relation + "?"
            question_answer_pair["label"] = object
        else:
            if object_type != "PERSON":
                first_pair = next(iter(topic.items()))
                if first_pair[1] != "revolution":
                    interrogative = "Which " + first_pair[1]
                else:
                    interrogative = "Which revolution or war"
            question_answer_pair["question"] = interrogative + " is " + " " + relation + " " + object + "?"
            question_answer_pair["label"] = subject
            
    elif subject_doc[0].tag_ == "VBD" and subject_doc[-1].tag_ not in ["NN", "NNP", "NNPS", "NNS"]:
        if not query_subject:
            question_answer_pair["question"] = interrogative + " did " + subject + " " 
            for token in subject_doc:
                if token.tag_ == "VBD":
                    question_answer_pair["question"] += token.lemma_ + " "
                else:
                    question_answer_pair["question"] += token.text + " "
            question_answer_pair["question"] = question_answer_pair["question"][:-1] + "?"
            question_answer_pair["label"] = object
        else:
            if object_type != "PERSON":
                first_pair = next(iter(topic.items()))
                if first_pair[1] != "revolution":
                    interrogative = "Which " + first_pair[1]
                else:
                    interrogative = "Which revolution or war"
            question_answer_pair["question"] = interrogative + " " + relation + " " + object + "?"
            question_answer_pair["label"] = subject

    elif (subject_doc[0].tag_ == "VB" or subject_doc[0].tag_ == "VBZ") and subject_doc[-1].tag_ not in ["NN", "NNP", "NNPS", "NNS"]:
        if not query_subject:
            question_answer_pair["question"] = interrogative + " does " + subject + " "
            for token in subject_doc:
                if token.tag_ == "VBZ":
                    question_answer_pair["question"] += token.lemma_ + " "
                else:
                    question_answer_pair["question"] += token.text + " "
            question_answer_pair["question"] = question_answer_pair["question"][:-1] + "?"
            question_answer_pair["label"] = object
        else:
            if object_type != "PERSON":
                first_pair = next(iter(topic.items()))
                if first_pair[1] != "revolution":
                    interrogative = "Which " + first_pair[1]
                else:
                    interrogative = "Which revolution or war"
            question_answer_pair["question"] = interrogative + " " + relation + " " + object + "?"
            question_answer_pair["label"] = subject

    elif (subject_doc[-1].tag_ == "NN" or subject_doc[-1].tag_ == "NNP") and subject_doc[0].tag_ not in ["VB", "VBZ", "VBD"]: 
        if not query_subject:
            question_answer_pair["question"] = interrogative + " is the " + relation + " of " + subject + "?"
            question_answer_pair["label"] = object
        else:
            first_pair = next(iter(topic.items()))
            if first_pair[1] == "human":
                question_answer_pair["question"] = interrogative + "se " + relation + " is " + object + "?"
            else:
                first_pair = next(iter(topic.items()))
                if first_pair[1] != "revolution":
                    interrogative = "Which " + first_pair[1]
                else:
                    interrogative = "Which revolution or war"
                question_answer_pair["question"] = interrogative + "'s " + relation + " is " + object + "?"
            question_answer_pair["label"] = subject
    else:
        return None
    
    return question_answer_pair            