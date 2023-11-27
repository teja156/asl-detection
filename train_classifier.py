import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

# lengths = [len(sublist) for sublist in data_dict['data']]
# if len(set(lengths)) != 1:
#     print("Inconsistent sublist lengths:", set(lengths))



incorrect = []
ind = 0
for i in data_dict['data']:
    if len(i)!=42:
        incorrect.append(ind)
    ind+=1

print("Number of inconsistent: ",len(incorrect))
# print("Incorrect indexes: ",incorrect)


data_dict['data'] = [entry for entry in data_dict['data'] if len(entry) == 42]

new_data_dict = {}
new_data_dict['data'] = []
new_data_dict['labels'] = []
ind=0
for i in data_dict['data']:
    if len(i)==42:
        new_data_dict['data'].append(i)
        new_data_dict['labels'].append(data_dict['labels'][ind])
    
    ind+=1

print("Removed inconsistent values")

data_dict = new_data_dict

print("Training...")
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()