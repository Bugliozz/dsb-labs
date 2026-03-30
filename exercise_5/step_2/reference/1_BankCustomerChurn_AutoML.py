import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, plot, draw
import h2o, os, pickle
from h2o.automl import H2OAutoML

def plot_ROC(axs, fpr, tpr, roc_auc, coluor='b', name='' ):
    axs.title.set_text('Receiver Operating Characteristic')
    axs.plot(fpr, tpr, coluor, label = name + ' AUC = %0.4f' % roc_auc)
    axs.legend(loc = 'lower right')
    axs.plot([0, 1], [0, 1],'r--')
    axs.set_xlim([0, 1])
    axs.set_ylim([0, 1])
    axs.set_ylabel('True Positive Rate')
    axs.set_xlabel('False Positive Rate')

data = pd.read_csv('Churn_Modelling.csv')
random_seed = 0
np.random.seed(random_seed)
msk = np.random.rand(len(data)) < 0.8
train = data[msk].reset_index()
validation = data[~msk].reset_index()

##############
### AUTOML ###
h2o.init(max_mem_size='1G')
h2o_train = h2o.H2OFrame(train)
h2o_validation = h2o.H2OFrame(validation)
# Identify predictors and response
x = h2o_train.columns
y = "Exited"
x.remove(y)

# For binary classification, response should be a factor
#h2o_train[y] = h2o_train[y].asfactor()
max_runtime_secs = round(60)
file_mame_autoML = str(random_seed) + "_autoML.pkl"
if not os.path.isfile(file_mame_autoML):
    aml = H2OAutoML(max_runtime_secs = max_runtime_secs, nfolds=5, seed = 1, project_name = "Churn_Modelling")
    aml.train(x = x, y = y,  training_frame = h2o_train)
    perf = aml.leader.model_performance()
    print(perf)
    # calculate the fpr and tpr for all thresholds of the classification
    probs = aml.leader.predict(h2o_validation.drop('Exited'))
    preds = probs[0].as_data_frame()
    preds = preds['predict'].tolist()
    pickle.dump(preds, open(file_mame_autoML, "wb")) 
else:
    preds = pickle.load(open(file_mame_autoML, "rb"))
    
fpr, tpr, threshold = metrics.roc_curve(validation['Exited'].tolist(), preds)
roc_auc = metrics.auc(fpr, tpr)

#############################
### Default Random Forest ###
t_rf = train.drop(['Exited','RowNumber','CustomerId','Surname'], axis=1)
v_rf = validation.drop(['Exited','RowNumber','CustomerId','Surname'], axis=1)
t_rf['Gender'] = t_rf['Gender'].replace({'Female': 1, 'Male': 0})
v_rf['Gender'] = v_rf['Gender'].replace({'Female': 1, 'Male': 0})
leg = preprocessing.LabelEncoder()
leg.fit(t_rf['Geography'])
t_rf['Geography'] = leg.transform(t_rf['Geography'])
v_rf['Geography'] = leg.transform(v_rf['Geography'])

clf = RandomForestClassifier(max_depth=6, random_state=123)
clf.fit(t_rf, train['Exited'])
preds_rf_p = clf.predict_proba(v_rf)[:,1]
fpr_rf_p, tpr_rf_p, threshold_rf_p = metrics.roc_curve(validation['Exited'].tolist(), preds_rf_p)
roc_auc_rf_p = metrics.auc(fpr_rf_p, tpr_rf_p)

########################
### EVALUATION  ROC  ###
fig, axs = plt.subplots(1)
plot_ROC(axs, fpr, tpr, roc_auc, coluor='g', name='AutoML')
plot_ROC(axs,fpr_rf_p, tpr_rf_p, roc_auc_rf_p, coluor='b', name='Random Forest')
draw()

#################################################
## Set the target for True Positive Rate ~= 80% ##
# AutoML
target_index = min(np.where( tpr > 0.8)[0])
target_threshold = threshold[target_index]
target_preds = preds > target_threshold
acc = metrics.accuracy_score(validation['Exited'].tolist(), target_preds)
print("False positive rate AutoML = " + str(round(fpr[target_index],3)) + "\n" +
      "Target threshold AutoML = " + str(round(target_threshold,3)) + "\n" + 
      "Accuracy AutoML = " + str(round(acc,4)))

disp = metrics.ConfusionMatrixDisplay.from_predictions(validation['Exited'].tolist(), target_preds)
disp.ax_.set_title('Confusion Matrix AutoML | Accuracy = ' + str(round(acc,3)) + '%')
draw()

# Random Forest
target_index_rf_p = min(np.where( tpr_rf_p > 0.8)[0])
target_threshold_rf_p = threshold_rf_p[target_index_rf_p]
target_preds_rf_p = preds_rf_p > target_threshold_rf_p
rf_acc = metrics.accuracy_score(validation['Exited'].tolist(), target_preds_rf_p)
print("False positive rate Random Forest = " + str(round(fpr_rf_p[target_index_rf_p],3)) + "\n" +
      "Target threshold Random Forest = " + str(round(target_threshold_rf_p,3)) + "\n" + 
      "Accuracy Random Forest = " + str(round(rf_acc,4)))
disp = metrics.ConfusionMatrixDisplay.from_predictions(validation['Exited'].tolist(), target_preds_rf_p)
disp.ax_.set_title('Confusion Matrix Random Forest | Accuracy = ' + str(round(rf_acc,3)) + '%')
draw()
show()

########################################################
## Try to improve classification via Cross-validation ##