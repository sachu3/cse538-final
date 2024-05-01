from conllu import parse_incr
import os

# PUT THE DEPENDENCY FILES IN ONE FOLDER AND PUT THE PATH TO THE FOLDER HERE
dep_files_path = 


### STRUCTURE OF DICTIONARIES FROM THE CONLLU FILES###
### VPC DICTIONARY ###
# {key : value}
# {sentence as string : [verbal particle, index of verbal particle, head of verbal particle, index of head]}

### VERBS WITH DIRECT OBJECTS DICTIONARY ###
# {key : value}
# {sentence as string : [direct object, index of direct object, head of direct object, index of head]}

vpc_sent_dict = {}
direct_obj_sent_dict = {}
sentence_count = 0
sentence_with_vpc = 0
sentence_with_direct_obj = 0
sentence_with_both = 0
tokens = 0
total_distance_vpc = 0
max_distance_vpc = 0
total_distance_obj = 0
max_distance_obj = 0

for file in os.listdir(dep_files_path): # builds the datset of sentences from the conllu files
    print(file)
    file = dep_files_path + file
    with open(file, "r", encoding="utf8") as f:
        for sentence in parse_incr(f): # parses each sentence in the file
            vpc_sent = False
            direct_obj_sent = False
            sentence_count += 1
            token_list = []
            for token in sentence:
                if token['head'] is not None:
                    if int(token['head']) == 0:
                        root_index = token['id'] # get the index and form of the root for getting direct object sentences
                        root_form = token['form']
                token_list.append(token['form'])
            tokens += len(token_list)
            for token in sentence:
                if token['deprel'] == "compound:prt": # this is the particle of a VPC, find the head and add it to the dataset
                    # print(sentence.metadata["text"])
                    vpc_sent_dict[sentence.metadata["text"]] = [token['form'], token['id'], token_list[int(token['head'] - 1)], token['head']]
                    sentence_with_vpc += 1
                    distance_vpc = abs(token['id'] - token['head'])
                    total_distance_vpc += distance_vpc # distance between verb and verb particle in current sentence
                    if distance_vpc > max_distance_vpc:
                        max_distance_vpc = distance_vpc
                    vpc_sent = True
                if token['deprel'] == "obj" and token['head'] == root_index: # this is the direct object of a sentence, if the head is the root of the sentence, add it to the dataset
                    direct_obj_sent_dict[sentence.metadata["text"]] = [token['form'], token['id'], token_list[root_index - 1], root_index]
                    sentence_with_direct_obj += 1
                    distance_obj = abs(token['id'] - token['head']) # distance between verb and object in current sentence
                    total_distance_obj += distance_obj
                    if distance_obj > max_distance_obj:
                        max_distance_obj = distance_obj 
                    direct_obj_sent = True
            if vpc_sent and direct_obj_sent:
                sentence_with_both += 1


### STRUCTURE OF LIST FOR IOB FILE DATA ###
# LIST CONSISTING OF: [[TOKENS OF SENTENCE], VPC_VERB, VPC_VERB_INDEX, VPC_PARTICLE, VPC_PARTICLE_INDEX]
## [TOKENS OF SENTENCE]: LIST OF THE TOKENS IN THE SENTENCE, THE IOB FILES DO NOT HAVE THE SENTENCES WRITTEN OUT AS STRINGS
## VPC_VERB: THE VERB OF THE VPC AS A STRING
## VPC_VERB_INDEX: THE INDEX OF THE VPC_VERB IN THE TOKENS OF THE SENTENCE AS AN INT
## VPC_PARTICLE: THE PARTICLE OF THE VPC AS A STRING
## VPC_PARTICLE_INDEX: THE INDEX OF THE VPC_PARTICLE IN THE TOKENS OF THE SENTENCE AS AN INT

# PUT THE IOB FILES INTO A FOLDER AND PUT THE PATH TO THE FOLDER HERE
file_path = 
 
vpc_sents = []

for file in os.listdir(file_path):
    print(file)
    file = file_path + file
    with open(file, "r", encoding="utf8") as file:
        vpc_verb = ""
        vpc_particle = ""
        sentence_toks = []
        sent_index = 0
        for line in file:
            if len(line.split()) > 0:
                line = line.split()
                token = line[0]
                sentence_toks.append(token)
                if line[1] == 'B-MWE_VPC': # this is the verb of a VPC
                    vpc_verb = token
                    vpc_verb_index = sent_index
                if line[1] == 'I-MWE_VPC': # this is the particle of a VPC
                    vpc_particle = token
                    vpc_particle_index = sent_index
                sent_index += 1
            else:
                if vpc_verb != "" and vpc_particle != "": # if this sentence has a VPC construction
                    vpc_sents.append([sentence_toks, vpc_verb, vpc_verb_index, vpc_particle, vpc_particle_index])
                    tokens += len(sentence_toks)
                    sentence_count += 1
                    sentence_with_vpc += 1
                    distance_vpc = abs(vpc_verb_index - vpc_particle_index)
                    if distance_vpc > max_distance_vpc:
                        max_distance_vpc = distance_vpc
                    total_distance_vpc += distance_vpc
                sentence_toks = []
                vpc_verb = ""
                vpc_particle = ""
                sent_index = 0

average_distance_vpc = total_distance_vpc / (len(vpc_sent_dict.keys()) + len(vpc_sents))
average_distance_obj = total_distance_obj / (len(direct_obj_sent_dict.keys()))

print("Total sentences: " + str(sentence_count))
print("Total tokens: " + str(tokens))
print("Sentences with VPCs: " + str(sentence_with_vpc))
print("Sentences with direct objects: " + str(sentence_with_direct_obj))
print("Sentences with both: " + str(sentence_with_both))
print("Average distance between verbal particle and verb: " + str(average_distance_vpc))
print("Max distance between verbal particle and verb: " + str(max_distance_vpc))
print("Average distance between direct objects and verb: " + str(average_distance_obj))
print("Max distance between direct objects and verb: " + str(max_distance_obj))



