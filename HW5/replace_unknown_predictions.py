import os

vgg_csv = "predictions-deepface-vgg.csv"
facenet_csv = "predictions-deepface-facenet.csv"

with open(vgg_csv, "r") as f:
    lines = f.readlines()

lines = lines[1:]
lines = [line[:-1] for line in lines]
vgg_predictions = [line.split(",") for line in lines]
vgg_predictions = [(int(line[0]), line[1]) for line in vgg_predictions]

print(vgg_predictions[:10])

with open(facenet_csv, "r") as f:
    lines = f.readlines()

lines = lines[1:]
lines = [line[:-1] for line in lines]
facenet_predictions = [line.split(",") for line in lines]
facenet_predictions = [(int(line[0]), line[1]) for line in facenet_predictions]

new_facenet_predictions = []
for i in range(len(facenet_predictions)):
    if facenet_predictions[i][1] == "=====UNKNOWN=====":
        new_facenet_predictions.append(vgg_predictions[i])
    else:
        new_facenet_predictions.append(facenet_predictions[i])

with open("predictions-deepface-new.csv", "w") as f:
    for id, cat in new_facenet_predictions:
        f.write(f"{id},{cat}\n")