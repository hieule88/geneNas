

# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# labels = [[1,2,3,4,1],\
#     [5,6,7,8,2],\
#     [9,10,11,12,3],\
#     [13,14,15,16,4]]

# preds = [[4,1,2,5],\
#     [5,0,1,0],\
#     [5,0,0,0],\
#     [3,0,0,3]]
# print('Preds before: ', preds)
# preds = [i for j in range(len(preds)) for i in preds[j][:labels[j][-1]] ]
# print('Preds after: ', preds)
# print('Labels before: ', labels)
# labels = [i for j in range(len(labels)) for i in labels[j][:labels[j][-1]] ]
# print('Labels after: ', labels )

# metrics = {}
# metrics['accuracy'] = accuracy_score(labels, preds)
# metrics['f1'] = f1_score(labels, preds, average='micro')
# metrics['recall'] = recall_score(labels, preds, average='micro')
# metrics['precision'] = precision_score(labels, preds, average='micro')

# print(metrics)

# b = {1:'a',2:'b',3:'c'}
# a = b

# a[4] = 'c'
# print(a)
# print(b)