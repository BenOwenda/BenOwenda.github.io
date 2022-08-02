from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_validate
import pickle

# ops = Options()
# ops.add_argument("--headless")
driver = webdriver.Chrome(executable_path="chromedriver.exe")#, options=ops)

driver.get("https://benowenda.github.io/")

drug_category = driver.find_elements(By.CLASS_NAME, "drug-category")
drug_price = driver.find_elements(By.CLASS_NAME, "drug-price")
drug_quantity = driver.find_elements(By.CLASS_NAME, "drug-quantity")
drug_score = driver.find_elements(By.CLASS_NAME, "drug-score")

table = []

for i in range(len(drug_category)-1):
    row = []
    row.append(int(drug_category[i].text))
    row.append(int(drug_price[i].text))
    row.append(int(drug_quantity[i].text))
    row.append(int(drug_score[i].text))

    table.append(row)

table_headers = ["category", "price", "quantity", "score"]

df = pd.DataFrame(table, columns=table_headers)

y = df["score"]
df = df.drop(["score"], axis=1)
X = df

classifier = KNN(n_neighbors=3)
classifier.fit(X, y)

#save the model
with open("webcrawler.pkl", 'wb') as file:
    pickle.dump(classifier, file)