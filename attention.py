from transformers import BertTokenizer, BertConfig, DistilBertTokenizer, DistilBertConfig, GPT2Tokenizer, GPT2Config
from transformers import AutoModel
import statistics
import parse_files

vpc_sent_dict, direct_obj_dict, vpc_sents = parse_files.run()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
model = AutoModel.from_config(config)
model = model.to("cuda")

count = 0
total_sents = len(vpc_sent_dict.keys())

attention_values_mean_max = []

for sentence in vpc_sent_dict.keys():
    attention_values_sent = []
    # input_text = "Respondents were provided the opportunity to select more than one race, and those who did were asked a follow-up question regarding which category best described their racial background."
    input_text = sentence
    verb_index = vpc_sent_dict[sentence][1] + 1
    particle_index = vpc_sent_dict[sentence][3] + 1
    tokenized_input = tokenizer(input_text, return_tensors='pt').to("cuda")
    output = model(**tokenized_input)
    attentions = output.attentions
    for layer in attentions:
        for x in range(11):
            attention_values_sent.append(layer[0][x][verb_index][particle_index].item())
    count += 1
    attention_values_mean_max.append([statistics.mean(attention_values_sent), max(attention_values_sent)])
    print(str(count) + "/" + str(total_sents))
     

print(statistics.mean(attention_values_sent))
print(max(attention_values_sent))
print(statistics.median(attention_values_sent))
