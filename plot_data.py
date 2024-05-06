### ~~CODE INFORMATION~~
### Team Members: Xiaoqun Liu, Steven Chu, John Idogun
#
### Description of Code: This file uses the CSV files created from attention_new.py to plot the average values found by each transformer.
### in a bar graph. 
#
### Class Concepts: This file visualizes how the attention of the verbs to its particles or objects looks. With the 
### graphs, we will try to make conclusions about the semantics [2.Semantics] of the constructions for the models.
#
### System used: Windows 10 (Python 3.11.3)

from matplotlib import pyplot as plt
from pathlib import Path
import statistics

path = Path("attention_results") 

model_names = [
                "Bert Base (Direct Objects)",
               "Bert Base (VPCs)", 
               "DistilBert (Direct Objects)", 
               "DistilBert (VPCs)",
                "GPT-2 (Direct Objects)",
                "GPT-2 (VPCs)", 
                "GPT-2 Medium (Direct Objects)", 
                "GPT-2 Medium (VPCs)"
                ] # model names for the plot titles

model_index = 0
for file in path.iterdir():
    mean_vals = []  # hold the average values per sentence to calculate overall mean later
    max_vals = []
    median_vals = []
    with open(file, 'r') as f:
        lines = f.readlines()
        lines.pop(0) # get rid of mean, max, median row at beginning
    for line in lines:
        line = line.split(",")
        mean_vals.append(float(line[0]))
        max_vals.append(float(line[1]))
        median_vals.append(float(line[2]))
    average_mean = statistics.mean(mean_vals)
    average_max = statistics.mean(max_vals)
    average_median = statistics.mean(median_vals)
    x_vals = ["Average Mean", "Average Max", "Average Median"]
    y_vals = [average_mean, average_max, average_median]
    
    plt.bar(x_vals, y_vals, width=.4, color='blue')
    plt.xlabel("Average Values")
    plt.ylabel("Attention values")
    plt.title(model_names[model_index])
    if model_index <= 3:
        plt.ylim(top=.16)
    else:
        plt.ylim(top=.015)
    model_index += 1
    plt.show()
