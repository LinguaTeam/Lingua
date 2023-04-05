import gradio as gr
import pandas as pd
import numpy as np
from simpletransformers.classification import ClassificationModel

def auth(username, password):
    if username == "Lingua" and password == "password":
        return True
    else:
        return False

def predict(texts):
    model_path = "bert_model"
    model = ClassificationModel('bert', model_path, use_cuda=False)
    predictions, _ = model.predict(texts)
    return [sayidan_sonuca(prediction) for prediction in predictions]

def sayidan_sonuca(sayi):
    if sayi == 4:
        return 'OTHER'
    elif sayi == 1:
        return 'RACIST'
    elif sayi == 0:
        return 'INSULT'
    elif sayi == 3:
        return 'PROFANITY'
    elif sayi == 2:
        return 'SEXIST'


def get_file(file):
    output_file = "output_Lingua.csv"
    df = pd.read_csv(file.name, sep="|")
    
    texts = df["text"].tolist()
    targets = predict(texts)
    
    df["target"] = targets
    df["offensive"] = df["target"].apply(lambda x: 1 if x != "OTHER" else 0)
    df = df.reindex(columns=['id', 'text', 'offensive', 'target'])
    df.to_csv(output_file, index=False, sep="|")
    
    return output_file


iface = gr.Interface(get_file, "file", "file")
iface.launch(share=True, auth=auth, debug=True)