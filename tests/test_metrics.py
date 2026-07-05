from src.evaluation.metrics import Evaluation

metric = Evaluation()

##################################################

gt = [

1,1,1,0,0,0,1,1,0,0

]

pred = [

1,0,1,0,0,1,1,0,0,0

]

##################################################

P = metric.precision(pred,gt)

R = metric.recall(pred,gt)

F = metric.fscore(P,R)

##################################################

print()

print("Precision :",round(P,4))

print("Recall    :",round(R,4))

print("F-score   :",round(F,4))