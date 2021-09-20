from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

model = KeyedVectors.load("word2vec.wv", mmap='r')
areas = ["toy", "machinery", "elevator", "explosive", "medical", "vehicle", "aviation",
         "biometric", "identification", "traffic", "water", "gas", "heating", "electricity",
         "education", "employment", "hiring", "credit", "lending", "emergency",
         "enforcement", "polygraph", "criminal", "profiling", "immigration", "asylum", "justice",
         "deepfake", "chatbot"]
values = [["transparency", "fairness", "discrimination"], ["accuracy", "robustness", "cybersecurity"]]

for value in values:
    fig = plt.figure(figsize=(10, 10))
    sim = [[model.similarity(area, v) for area in areas] for v in value]
    ax = plt.axes(projection='3d')
    ax.scatter(sim[0], sim[1], sim[2], c='r', marker='o')
    for i in range(len(areas)):
        ax.text(sim[0][i], sim[1][i], sim[2][i], areas[i])
    ax.set_xlabel(value[0])
    ax.set_ylabel(value[1])
    ax.set_zlabel(value[2])
    ax.view_init(10, 150)
    plt.savefig(" ".join(value) + ".png")