import pickle
import pandas as pd

#load model
with open("webcrawler.pkl", 'rb') as file:
    classifier = pickle.load(file)

data = [[1,140,21]]
df = pd.DataFrame(data)
prediction = classifier.predict(data)
print(prediction)
