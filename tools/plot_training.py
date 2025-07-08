import matplotlib.pyplot as plt
import  re
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('log_path', type=str,
                    help='path to the log.txt file')

args = parser.parse_args()

with open(args.log_path) as f:
    text = f.read()
    
    matches = re.findall(r'"name":\s*"([^"]+)",\s*"value":\s*([0-9.eE+-]+)', text)


collected_values =  {
    "Accuracy": [],
    "F1_score": [],
    "FP": [],
    "FN": [],
    "count": []
}

count = 0
for name, value in matches:
    count += 1
    
    collected_values[name].append(value)
    
    if count % 4 == 0:
        collected_values["count"].append(count / 4 * 3)
        
collected_values["Accuracy"] = [float(x) for x in collected_values["Accuracy"]]
collected_values["F1_score"] = [float(x) for x in collected_values["F1_score"]]
collected_values["FP"] = [float(x) for x in collected_values["FP"]]
collected_values["FN"] = [float(x) for x in collected_values["FN"]] 
    
fig, ax = plt.subplots(nrows=2, ncols=2)
plt.setp(ax, xticks=collected_values["count"])


ax[0][0].plot(collected_values["count"], collected_values["Accuracy"])
ax[0][0].title.set_text('Accuracy')

ax[0][1].plot(collected_values["count"], collected_values["F1_score"])
ax[0][1].title.set_text('F1_score')

ax[1][0].plot(collected_values["count"], collected_values["FP"])
ax[1][0].title.set_text('FP')

ax[1][1].plot(collected_values["count"], collected_values["FN"])
ax[1][1].title.set_text('FN')

plt.show()
