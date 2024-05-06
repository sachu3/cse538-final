### ~~CODE INFORMATION~~
### Team Members: Xiaoqun Liu, Steven Chu, John Idogun
#
### Description of Code: This file uses the data parsed from parse_files.py to get the attention values from the four models for analysis.
### for each model (BERT, distilBERT, GPT-2, GPT-2 Medium), tokenize each example, keep track of the verb and its particle, get the attention
### values of the verb to its particle. Do the same for the direct object data. Write all results, the mean, max, median attention values per sentence
### to separate CSV files.  
#
### Class Concepts: For each sentence, we tokenized [1.Syntax] each example in order to run them through each model. We then fed the tokenized
### sentences to each transformer [3.Transformer Models] and extracted the attention values. We then looked at specific attention values to later see
### how the verbs and their particles or objects are related semantically [2.Semantics]
#
### System used: Windows 10 (Python 3.11.3), Google Colab.

import logging
import statistics
from pathlib import Path

import torch
from transformers import (
    AutoModel,
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    GPT2Config,
    GPT2TokenizerFast,
)

from parse_files2 import run as parse_files_run

# set up logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s'
)


def extract_attention_bert(vpc_sent_dict: dict, device: str, model_name: str) -> list:
    """
    Extracts attention values from a BERT model for a set of sentences.

    Args:
        `vpc_sent_dict (dict)`: Dictionary with sentences as keys and lists of verb and particle indices as values.
        `device (str)`: Device to run the model on ('cpu', 'cuda').
        `model_name (str)`: Name of the pre-trained BERT model to use.

    Returns:
        `list`: List of lists, each containing the mean, max, and median attention values for a sentence.

    The function performs the following steps:
        - Initializes the tokenizer and the BERT model using the provided model name.
        - Sets the model to evaluation mode and moves it to the specified device.
        - Initializes an empty list to store the attention values for each sentence.
        - Iterates over each sentence in the input dictionary.
        - Tokenizes each sentence and feeds it to the model to obtain the attention values.
        - Extracts the attention values corresponding to the verb and particle indices in the sentence.
        - Calculates the mean, max, and median of these attention values and appends them to the list.
        - Logs the progress of the processing.
        - Returns the list of attention values.
    """
    tokenizer = (
        BertTokenizer.from_pretrained(model_name)
        if 'distil' not in model_name
        else DistilBertTokenizer.from_pretrained(model_name)
    )
    config = (
        BertConfig.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=True
        )
        if 'distil' not in model_name
        else DistilBertConfig.from_pretrained(
            model_name, output_attentions=True, output_hidden_states=True
        )
    )
    model = AutoModel.from_config(config)
    model = model.to(device)
    model.eval()

    attention_values_mean_max_median = []
    count = 0
    total_sents = len(vpc_sent_dict.keys())

    for sentence in vpc_sent_dict.keys():
        attention_values_sent = []
        input_text = sentence
        verb_form = vpc_sent_dict[sentence][2] # the verb form as a string
        particle_form = vpc_sent_dict[sentence][0] # the particle form as a string
        logging.info(f'Verb_Form: {verb_form}')
        logging.info(f'Particle_Form: {particle_form}')
        tokenized_sent = tokenizer.encode(input_text)
        tokenized_input = tokenizer(input_text, return_tensors='pt').to(device)
        tokenized_sentence = tokenizer.convert_ids_to_tokens(tokenized_sent) # get the tokenized sentence as strings
        logging.info(f'Tokenized sentence: {tokenized_sentence}')
        output = model(**tokenized_input)
        attentions = output.attentions
        try:

            if tokenized_sentence[0] == verb_form:
                verb_form = verb_form 
            # Check if the verb form is in the tokenized sentence
            elif verb_form not in tokenized_sentence:
                verb_form = verb_form # bert tokenizer does not add space marker

            logging.info(f'Verb form: {verb_form}')
            # Prepare particle form based on proximity to dash
        
            verb_index = tokenized_sentence.index(verb_form)

            logging.info(f'Particle form: {particle_form}')

            particle_index = tokenized_sentence.index(particle_form)

            for layer in attentions: # find all of the values of attention from the verb to its particle (or the verb to its direct object)
                for x in range(len(layer[0])):
                    attention_values_sent.append(
                        layer[0][x][verb_index][particle_index].item()
                    )

            attention_values_mean_max_median.append(
                [
                    statistics.mean(attention_values_sent),
                    max(attention_values_sent),
                    statistics.median(attention_values_sent),
                ]
            )
        except ValueError as e: # couldn't find one of the tokens, this usually happens when a verb form is split by the tokenizer
            logging.error(
                f'Skipping sentence due to no matching token: {sentence}, error: {e}'
            )
            continue

        count += 1
        logging.info(
            f'{model_name.replace("-", " ").title()} processed {count}/{total_sents}'
        )

    return attention_values_mean_max_median


def extract_attention_gpt2(vpc_sent_dict: dict, device: str, model_name: str) -> list:
    """
    Extracts attention values from a GPT-2 model for a set of sentences.

    Args:
        `vpc_sent_dict (dict)`: Dictionary with sentences as keys and lists of verb and particle indices as values.
        `device (str)`: Device to run the model on ('cpu', 'cuda').
        `model_name (str)`: Name of the pre-trained GPT-2 model to use.

    Returns:
        `list`: List of lists, each containing the mean, max, and median attention values for a sentence.

    The function performs the following steps:
        - Initializes the tokenizer and the GPT-2 model using the provided model name.
        - Sets the model to evaluation mode and moves it to the specified device.
        - Initializes an empty list to store the attention values for each sentence.
        - Iterates over each sentence in the input dictionary.
        - Tokenizes each sentence and feeds it to the model to obtain the attention values.
        - Extracts the attention values corresponding to the verb and particle indices in the sentence.
        - Calculates the mean, max, and median of these attention values and appends them to the list.
        - Logs the progress of the processing.
        - Returns the list of attention values.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    config = GPT2Config.from_pretrained(
        model_name, output_attentions=True, output_hidden_states=True
    )
    model = AutoModel.from_pretrained(model_name, config=config)
    model = model.to(device)
    model.eval()

    attention_values_mean_max_median = []
    count = 0
    total_sents = len(vpc_sent_dict.keys())

    for sentence, values in vpc_sent_dict.items():
        input_text = sentence
        inputs = tokenizer(input_text, return_tensors='pt', add_special_tokens=True).to(
            device
        )
        outputs = model(**inputs)
        attentions = outputs.attentions

        verb_form = values[2]
        particle_form = values[0]

        tokenized_sentence = tokenizer.tokenize(input_text)
        logging.info(f'Tokenized sentence: {tokenized_sentence}')
        # Adjust the verb form, check if it should have the 'Ġ' prefix
        if tokenized_sentence[0] == verb_form:
            verb_form = verb_form  # No 'Ġ' if verb is the first word in the sentence
        # Check if the verb form is in the tokenized sentence
        elif verb_form not in tokenized_sentence:
            verb_form = 'Ġ' + verb_form if not verb_form.startswith('Ġ') else verb_form

        logging.info(f'Verb form: {verb_form}')
        # Prepare particle form based on proximity to dash
        try:
            verb_index = tokenized_sentence.index(verb_form)
            # Check if there is a '-' immediately after the verb in the tokenized sentence
            if (
                verb_index + 1 < len(tokenized_sentence)
                and tokenized_sentence[verb_index + 1] == '-'
            ):
                particle_form = particle_form  # Use without 'Ġ' if '-' follows the verb
            else:
                particle_form = (
                    'Ġ' + particle_form
                    if not particle_form.startswith('Ġ')
                    else particle_form
                )

            logging.info(f'Particle form: {particle_form}')

            particle_index = tokenized_sentence.index(particle_form)

            attention_values_sent = []
            for layer in attentions:
                for x in range(len(layer[0])): # get the attention values from the verb to its particle, or the direct object to its object
                    attention_values_sent.append(
                        layer[0][x][verb_index][particle_index].item()
                    )

            attention_values_mean_max_median.append(
                [
                    statistics.mean(attention_values_sent),
                    max(attention_values_sent),
                    statistics.median(attention_values_sent),
                ]
            )
        except ValueError as e:
            logging.error(
                f'Skipping sentence due to no matching token: {sentence}, error: {e}'
            )
            continue

        count += 1
        logging.info(f'{model_name.upper()} processed {count}/{total_sents}')

    return attention_values_mean_max_median


def write_attention_values_to_csv(attention_values: list, output_file: Path):
    """
    Writes the attention values to a CSV file.

    Args:
        `attention_values (list)`: List of lists, each containing the mean, max, and median attention values for a sentence.
        `output_file (Path)`: Path to the output CSV file.

    The function writes the attention values to the specified CSV file in the following format:
        mean,max,median
        value1,value2,value3
        value4,value5,value6
        ...
    """
    with output_file.open('w') as f:
        f.write('mean,max,median\n')
        for row in attention_values:
            f.write(','.join(map(str, row)) + '\n')


def main():
    """
    Main function to extract attention values from BERT and GPT-2 models.

    The function performs the following steps:
        - Sets the seed for PyTorch for reproducibility.
        - Determines the device to run the models on based on the availability of CUDA.
        - Parses the input files and extracts the sentences and their corresponding verb and particle indices.
        - Creates a directory to store the output files if it doesn't exist.
        - Extracts attention values from the 'bert-base-uncased' model and writes them to a CSV file, one for VPCs and one for direct objects.
        - Extracts attention values from the 'distilbert-base-uncased' model and writes them to a CSV file, one for VPCs and one for direct objects.
        - Extracts attention values from the 'gpt2' model and writes them to a CSV file, one for VPCs and one for direct objects.
        - Extracts attention values from the 'gpt2-medium' model and writes them to a CSV file, one for VPCs and one for direct objects.
    """
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vpc_sent_dict, direct_obj_dict, vpc_sents = parse_files_run()
    output_dir = Path('attention_results')
    output_dir.mkdir(exist_ok=True)

    logging.info('Extracting with BERT base...')
    bert_results = extract_attention_bert(vpc_sent_dict, device, 'bert-base-uncased')
    # logging.info(f'BERT base results: {bert_results}')
    bert_base_output = output_dir / 'bert_base_attention_values.csv'
    write_attention_values_to_csv(bert_results, bert_base_output)

    logging.info('Extracting with BERT base Direct_obj...')
    bert_results = extract_attention_bert(direct_obj_dict, device, 'bert-base-uncased')
    # logging.info(f'BERT base results: {bert_results}')
    bert_base_output = output_dir / 'bert_base_attention_values_DO.csv'
    write_attention_values_to_csv(bert_results, bert_base_output)

    logging.info('Extracting with BERT distil...')
    bert_distil_results = extract_attention_bert(
        vpc_sent_dict, device, 'distilbert-base-uncased'
    )
    # logging.info(f'BERT distil results: {bert_distil_results}')
    bert_distil_output = output_dir / 'bert_distil_attention_values.csv'
    write_attention_values_to_csv(bert_distil_results, bert_distil_output)

    logging.info('Extracting with BERT distil Direct_obj...')
    bert_distil_results = extract_attention_bert(
        direct_obj_dict, device, 'distilbert-base-uncased'
    )
    # logging.info(f'BERT distil results: {bert_distil_results}')
    bert_distil_output = output_dir / 'bert_distil_attention_values_DO.csv'
    write_attention_values_to_csv(bert_distil_results, bert_distil_output)

    logging.info('Extracting with GPT-2...')
    gpt2_results = extract_attention_gpt2(vpc_sent_dict, device, 'gpt2')
    # logging.info(f'GPT-2 results: {gpt2_results}')
    gpt2_output = output_dir / 'gpt2_attention_values.csv'
    write_attention_values_to_csv(gpt2_results, gpt2_output)

    logging.info('Extracting with GPT-2 Direct_obj...')
    gpt2_results = extract_attention_gpt2(direct_obj_dict, device, 'gpt2')
    # logging.info(f'GPT-2 results: {gpt2_results}')
    gpt2_output = output_dir / 'gpt2_attention_values_DO.csv'
    write_attention_values_to_csv(gpt2_results, gpt2_output)

    logging.info('Extracting with GPT-2 medium...')
    gpt2_medium_results = extract_attention_gpt2(vpc_sent_dict, device, 'gpt2-medium')
    # logging.info(f'GPT-2 medium results: {gpt2_medium_results}')
    gpt2_medium_output = output_dir / 'gpt2_medium_attention_values.csv'
    write_attention_values_to_csv(gpt2_medium_results, gpt2_medium_output)

    logging.info('Extracting with GPT-2 medium Direct_obj...')
    gpt2_medium_results = extract_attention_gpt2(direct_obj_dict, device, 'gpt2-medium')
    # logging.info(f'GPT-2 medium results: {gpt2_medium_results}')
    gpt2_medium_output = output_dir / 'gpt2_medium_attention_values_DO.csv'
    write_attention_values_to_csv(gpt2_medium_results, gpt2_medium_output)


if __name__ == '__main__':
    main()
