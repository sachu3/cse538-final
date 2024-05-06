### ~~CODE INFORMATION~~
### Team Members: Xiaoqun Liu, Steven Chu, John Idogun
#
### Description of Code: This file parses the files for the sentence data we need to do our analysis.
### It builds two dictionaries that hold the data for analysis: one is a dictionary with info on sentences 
### that have VPCs, and the other is a dictionary with info on sentences that have direct objects. After it is
### done gathering data, it prints out general information about the data (print statements at end of file).
#
### Class Concepts: In looping through the files (lines 53-90), Syntacic [1. Syntax] concepts of dependencies and heads are noted in the conllu files.
### We use this information to tell us what to look for when we do our analysis with the attention of the models. This information is also present
### in the IOB files, but the dependencies are not explicitly written into the data as they are in conllu files. We infer the relationship through
### the data in the sentence itself. 
#
### System used: Windows 10 (Python 3.11.3), Google Colab. Packages needed: conllu


from pathlib import Path

from conllu import parse_incr


def run():
    # PUT THE DEPENDENCY FILES IN ONE FOLDER AND PUT THE PATH TO THE FOLDER HERE
    dep_files_path = Path('dep_files')

    ### STRUCTURE OF DICTIONARIES FROM THE CONLLU FILES###
    ### VPC DICTIONARY ###
    # {key : value}
    # {sentence as string : [verbal particle, index of verbal particle, head of verbal particle, index of head]}

    ### VERBS WITH DIRECT OBJECTS DICTIONARY ###
    # {key : value}
    # {sentence as string : [direct object, index of direct object, head of direct object, index of head]}

    vpc_sent_dict = {}
    direct_obj_sent_dict = {}
    sentence_count = 0  # total count of sentences
    sentence_with_vpc = 0  # count of sentences that have 
    sentence_with_direct_obj = 0  # count of sentences with a direct object(s)
    sentence_with_both = 0  # count of sentences with a direct object AND a VPC
    tokens = 0  # total number of tokens (according to the conllu files and iob files)
    total_distance_vpc = 0  # value used to calculate average distance between verb and its particle
    max_distance_vpc = 0  # find the max distance between a verb and its particle (from our data, it's 16 tokens)
    total_distance_obj = 0  # value used to calculate average distance between verb and its object
    max_distance_obj = 0  # find the max distance between 

    for file_path in dep_files_path.iterdir():
        print(file_path)
        with open(file_path, 'r', encoding='utf8') as f:
            # parses each sentence in the file
            for sentence in parse_incr(f):
                vpc_sent = False
                direct_obj_sent = False
                sentence_count += 1
                token_list = []
                for token in sentence: # get the tokens of the sentence, and find the root of the sentence
                    if token['head'] is not None:
                        if int(token['head']) == 0:
                            # get the index and form of the root for getting direct object sentences
                            root_index = token['id']
                            root_form = token['form']
                    if '-' not in str(token['id']): # this is to deal with contractions, they mess up the token list because they are listed together and broken up (ie. ["we're", "we", "'re"])
                        token_list.append(token['form'])
                tokens += len(token_list)
                for token in sentence: # find VPCs or direct objs
                    # this is the particle of a VPC, find the head and add it to the dataset
                    if token['deprel'] == 'compound:prt':
                        # print(sentence.metadata["text"])
                        vpc_sent_dict[sentence.metadata['text']] = [
                            token['form'],
                            token['id'],
                            token_list[int(token['head'] - 1)],
                            token['head'],
                        ] # add the info from the sentence data to the dictionary
                        sentence_with_vpc += 1
                        distance_vpc = abs(token['id'] - token['head'])
                        # distance between verb and verb particle in current sentence
                        total_distance_vpc += distance_vpc
                        if distance_vpc > max_distance_vpc:
                            max_distance_vpc = distance_vpc
                        vpc_sent = True
                    # this is the direct object of a sentence, if the head is the root of the sentence, add it to the dataset
                    if token['deprel'] == 'obj' and token['head'] == root_index:
                        direct_obj_sent_dict[sentence.metadata['text']] = [
                            token['form'],
                            token['id'],
                            token_list[root_index - 1],
                            root_index,
                        ]
                        sentence_with_direct_obj += 1
                        distance_obj = abs(
                            token['id'] - token['head']
                        )  # distance between verb and object in current sentence
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
    iob_files_path = Path('iob_files')

    vpc_sents = [] # list described above for holding data from IOB files

    for file_path in iob_files_path.iterdir():
        print(file_path)
        # file = file_path + file
        with open(file_path, 'r', encoding='utf8') as f:
            vpc_verb = ''
            vpc_particle = ''
            sentence_toks = []
            sent_index = 0
            for line in f:
                if len(line.split()) > 0:
                    line = line.split()
                    token = line[0]
                    sentence_toks.append(token)
                    if line[1] == 'B-MWE_VPC':  # this is the verb of a VPC
                        vpc_verb = token
                        vpc_verb_index = sent_index
                    if line[1] == 'I-MWE_VPC':  # this is the particle of a VPC
                        vpc_particle = token
                        vpc_particle_index = sent_index
                    sent_index += 1
                else:
                    # if this sentence has a VPC construction
                    if vpc_verb != '' and vpc_particle != '':
                        vpc_sents.append(
                            [
                                sentence_toks,
                                vpc_verb,
                                vpc_verb_index,
                                vpc_particle,
                                vpc_particle_index,
                            ]
                        ) # add the data from the sentence to the list
                        tokens += len(sentence_toks)
                        sentence_count += 1
                        sentence_with_vpc += 1
                        distance_vpc = abs(vpc_verb_index - vpc_particle_index)
                        if distance_vpc > max_distance_vpc:
                            max_distance_vpc = distance_vpc
                        total_distance_vpc += distance_vpc
                    sentence_toks = []
                    vpc_verb = ''
                    vpc_particle = ''
                    sent_index = 0

    average_distance_vpc = total_distance_vpc / (
        len(vpc_sent_dict.keys()) + len(vpc_sents)
    )
    average_distance_obj = total_distance_obj / (len(direct_obj_sent_dict.keys()))

    for sentence in vpc_sents:
        sent_string = " ".join(sentence[0])
        vpc_sent_dict[sent_string] = [sentence[3], sentence[4], sentence[1], sentence[2]]
        # add sentences from IOB files to dictionary in same format

    print(f'Total sentences: {str(sentence_count)}')
    print(f'Total tokens: {str(tokens)}')
    print(f'Sentences with VPCs: {str(sentence_with_vpc)}')
    print(f'Sentences with direct objects: {str(sentence_with_direct_obj)}')
    print(f'Sentences with both: {str(sentence_with_both)}')
    print(
        f'Average distance between verbal particle and verb: {str(average_distance_vpc)}'
    )
    print(f'Max distance between verbal particle and verb: {str(max_distance_vpc)}')
    print(
        f'Average distance between direct objects and verb: {str(average_distance_obj)}'
    )
    print(f'Max distance between direct objects and verb: {str(max_distance_obj)}')
    return vpc_sent_dict, direct_obj_sent_dict, vpc_sents

