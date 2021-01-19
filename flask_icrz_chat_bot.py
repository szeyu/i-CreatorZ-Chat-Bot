
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
#import tensorflow as tf
import random
from tensorflow.python.framework import ops
import pickle
from datetime import datetime 
import pytz 



# import our chat-bot intents file
import json
with open('json file/intents.json') as json_data:
    data = json.load(json_data)
    

from flask import Flask, render_template, request
app = Flask(__name__, template_folder='templates')

messages = []
reply = []
message_time = []
reply_time = []

    
    
## Data Set
try:
    with open("data_icrzBot.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
    
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    
    # loop through each sentence in our intents patterns
    for intent in data['intents']:
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            wrds = nltk.word_tokenize(pattern)
            # add to our words list
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])
            
        # add to our labels list
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
    
    #print(docs_x)
    #print(docs_y)
    
    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))
    
    labels = sorted(labels)
    
    #print(words)
    #print(labels)
    
    training = []
    output = []
    
    out_empty = [0 for _ in range(len(labels))]
    
    
    
    for x, doc in enumerate(docs_x):
        bag = []
        
        wrds = [stemmer.stem(w.lower()) for w in doc if w not in "?"]
        
        #print(doc)
        #print(wrds)
        
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
            
        
        #print(bag)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)
        
        
    training = np.array(training)
    output = np.array(output)
        
    #print(training)
    #print("#####################################################################")
    #print(output)
    
    with open("data_icrzBot.pickle", "wb") as f:
        pickle.dump((words, labels, training, output),f)
        
        
    
#Tf learn

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")

net = tflearn.regression(net, 
                         optimizer='adam', 
                         loss='categorical_crossentropy', 
                         learning_rate=0.001)

model = tflearn.DNN(net)

try:
    model.load("model_icrzBot.tflearn")

except:
    model.fit(training, output, 
              n_epoch=1000, 
              batch_size=8, 
              show_metric=True)
    
    model.save("model_icrzBot.tflearn")
    
    
    
def bag_of_words(sentence, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(sentence)
    s_words = [stemmer.stem(word.lower()) for word in s_words if word not in "?"]
    
    for s in s_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                
    return np.array(bag)


def reply_message(user_input):
    result = model.predict([bag_of_words(user_input, words)])[0]  #since this is list of list
    result_index = np.argmax(result)
    tag = labels[result_index]
    
    if result[result_index] > 0.7:  # 70% confidence
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                break
    
        reply.append(random.choice(responses))
    else:
        reply.append("I didn't get that, try again.")
    reply_time.append(str(datetime.now(pytz.timezone('Asia/Kuala_Lumpur')).strftime("%H:%M")))
    



@app.route('/', methods=['GET', 'POST'])
def chat():
    #print('message was received!!!')
    if request.method == "POST":
        message = request.form["chatbox"]
        messages.append(message)
        message_time.append(str(datetime.now(pytz.timezone('Asia/Kuala_Lumpur')).strftime("%H:%M")))
        reply_message(message)
        return render_template("icrzBotIndex.html", messages=messages, reply=reply, message_time=message_time, reply_time=reply_time)
    
    else:
        return render_template("icrzBotIndex.html", messages=messages, reply=reply)


if __name__ == "__main__":
    app.run(threaded=True, port=5000)



#%%
