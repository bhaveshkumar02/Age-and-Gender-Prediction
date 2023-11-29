import os
import cv2
from deepface import DeepFace
import pandas as pd

# img = cv2.imread("images/Jacob.jpg")
# results = DeepFace.analyze(img, actions=("gender", "age", "race", "emotion"))
# print(results)

data = {
    "Name": [],
    "Age": [],
    "Gender": [],
    "Race": []
}

for file in os.listdir("images"):
    result = DeepFace.analyze(cv2.imread(f"images\{file}"), actions=("age", "gender", "race"))
    data["Name"].append(file.split(".")[0])
    data["Age"].append(result[0]["age"])
    data["Gender"].append(result[0]["dominant_gender"])
    data["Race"].append(result[0]["dominant_race"])

df = pd.DataFrame(data)
print(df)

df.to_csv("people.csv")