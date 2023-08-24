import os

locations = ["data/graphs/training.g", "data/graphs/test.g", "data/graphs/val.g"]

enriched_graphs = open('data/graphs/BPI12_graph.g', 'r').read()
graphs = enriched_graphs.split('XP\n')
graphs.remove('')
percentage_train = 0.65
percentage_test = 0.25
number_train_graphs = int(len(graphs)*percentage_train)
number_test_graphs = int(len(graphs)*percentage_test)
number_val_graphs = len(graphs)-number_test_graphs-number_train_graphs
train = graphs[:number_train_graphs]
test = graphs[number_train_graphs+1:number_train_graphs+number_test_graphs]
val = graphs[-number_val_graphs:]
split = [train, test, val]

for i in range(0, len(locations)):
    if os.path.exists(locations[i]):
        os.remove(locations[i])
    with open(locations[i], 'w') as file:
        for spl in split[i]:
            file.write('XP\n')
            file.write(spl)

print('Train_size: ' + str(number_train_graphs))
print('Test_size: ' + str(number_test_graphs))
print('Val_size: ' + str(number_val_graphs))
