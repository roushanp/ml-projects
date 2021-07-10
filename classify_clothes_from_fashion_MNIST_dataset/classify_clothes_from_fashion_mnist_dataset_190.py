#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import gzip
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def showImage(data):
    some_article = data   # Selecting the image.
    some_article_image = some_article.reshape(28, 28) # Reshaping it to get the 28x28 pixels
    plt.imshow(some_article_image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()


# In[4]:


filePath_train_set =  '/cxldata/datasets/project/fashion-mnist/train-images-idx3-ubyte.gz'


# In[5]:


filePath_train_label = '/cxldata/datasets/project/fashion-mnist/train-labels-idx1-ubyte.gz'


# In[6]:


filePath_test_set = '/cxldata/datasets/project/fashion-mnist/t10k-images-idx3-ubyte.gz'


# In[7]:


filePath_test_label = '/cxldata/datasets/project/fashion-mnist/t10k-labels-idx1-ubyte.gz'


# In[8]:


with gzip.open(filePath_train_label, 'rb') as trainLbpath:
     trainLabel = np.frombuffer(trainLbpath.read(), dtype=np.uint8,
                               offset=8)
with gzip.open(filePath_train_set, 'rb') as trainSetpath:
     trainSet = np.frombuffer(trainSetpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(trainLabel), 784)

with gzip.open(filePath_test_label, 'rb') as testLbpath:
     testLabel = np.frombuffer(testLbpath.read(), dtype=np.uint8,
                               offset=8)

with gzip.open(filePath_test_set, 'rb') as testSetpath:
     testSet = np.frombuffer(testSetpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(testLabel), 784)


# In[9]:


X_train = trainSet
X_test = testSet
y_train = trainLabel
y_test = testLabel


# In[10]:


showImage(X_train[0])


# In[11]:


y_train[0]


# In[12]:


np.random.seed(42)


# In[13]:


shuffle_index = np.random.permutation(60000)


# In[14]:


X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# In[15]:


from sklearn.preprocessing import StandardScaler


# In[16]:


scaler = StandardScaler()


# In[17]:


X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))


# In[18]:


from sklearn.metrics import accuracy_score


# In[19]:


from sklearn.metrics import precision_score, recall_score


# In[20]:


from sklearn.metrics import f1_score


# In[21]:


from sklearn.linear_model import LogisticRegression


# In[22]:


# using Softmax Regression (multi-class classification problem)

log_clf = LogisticRegression(C=10, solver="lbfgs", multi_class="multinomial", random_state=42)

# 'C' is hyprparameter for regularizing L2
# 'lbfgs' is Byoden-Fletcher-Goldfarb-Shanno(BFGS) algorithm


# In[23]:


log_clf.fit(X_train_scaled, y_train)


# In[24]:


y_train_predict = log_clf.predict(X_train[0].reshape(1, -1))


# In[25]:


y_train[0] 

y_train_predict[0]

showImage(X_train[0])


# In[26]:


y_train_predict = log_clf.predict(X_train_scaled)


# In[27]:


log_accuracy = accuracy_score(y_train, y_train_predict)

log_precision = precision_score(y_train, y_train_predict, average='weighted')

log_recall = recall_score(y_train, y_train_predict, average='weighted')

log_f1_score = f1_score(y_train, y_train_predict, average='weighted')


# In[28]:


from sklearn.ensemble import RandomForestClassifier


# In[29]:


rnd_clf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)


# In[30]:


rnd_clf.fit(X_train, y_train)


# In[31]:


y_train_predict = rnd_clf.predict(X_train[0].reshape(1, -1))


# In[32]:


y_train[0] 

y_train_predict[0]

showImage(X_train[0])


# In[33]:


y_train_predict = rnd_clf.predict(X_train)


# In[34]:


rnd_accuracy = accuracy_score(y_train, y_train_predict)

rnd_precision = precision_score(y_train, y_train_predict, average='weighted')

rnd_recall = recall_score(y_train, y_train_predict, average='weighted')

rnd_f1_score = f1_score(y_train, y_train_predict, average='weighted')


# In[35]:


from sklearn.model_selection import cross_val_score, cross_val_predict


# In[36]:


from sklearn.metrics import confusion_matrix


# In[37]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[38]:


log_clf = LogisticRegression(C=10, solver="lbfgs", multi_class="multinomial", random_state=42)


# In[39]:


log_cv_scores = cross_val_score(log_clf, X_train_scaled, y_train, cv=3, scoring="accuracy") 
display_scores(log_cv_scores)


# In[40]:


log_cv_accuracy = log_cv_scores.mean()


# In[41]:


y_train_pred = cross_val_predict(log_clf, X_train_scaled, y_train, cv=3)


# In[42]:


confusion_matrix(y_train, y_train_pred)


# In[43]:


log_cv_precision = precision_score(y_train, y_train_pred, average='weighted')


# In[44]:


log_cv_recall = recall_score(y_train, y_train_pred, average='weighted')


# In[45]:


log_cv_f1_score = f1_score(y_train, y_train_pred, average='weighted')


# In[46]:


print(log_cv_accuracy)
print(log_cv_accuracy)
print(log_cv_recall)
print(log_cv_f1_score)


# In[47]:


rnd_clf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)


# In[48]:


rnd_cv_scores = cross_val_score(rnd_clf, X_train, y_train, cv=3, scoring="accuracy") 
display_scores(rnd_cv_scores)


# In[49]:


rnd_cv_accuracy = rnd_cv_scores.mean()


# In[50]:


y_train_pred = cross_val_predict(rnd_clf, X_train, y_train, cv=3)


# In[51]:


confusion_matrix(y_train, y_train_pred)


# In[52]:


rnd_cv_precision = precision_score(y_train, y_train_pred, average='weighted')


# In[53]:


rnd_cv_recall = recall_score(y_train, y_train_pred, average='weighted')


# In[54]:


rnd_cv_f1_score = f1_score(y_train, y_train_pred, average='weighted')


# In[55]:


print(rnd_cv_accuracy)
print(rnd_cv_accuracy)
print(rnd_cv_recall)
print(rnd_cv_f1_score)


# In[56]:


print("=== Softmax === ")
display_scores(log_cv_scores)
print("log_cv_accuracy:", log_cv_accuracy)
print("log_cv_precision:", log_cv_precision)
print("log_cv_recall:", log_cv_recall)
print("log_cv_f1_score:", log_cv_f1_score)

print("=== Random Forest === ")
display_scores(rnd_cv_scores)
print("rnd_cv_accuracy:", rnd_cv_accuracy)
print("rnd_cv_precision:", rnd_cv_precision)
print("rnd_cv_recall :", rnd_cv_recall )
print("rnd_cv_f1_score:", rnd_cv_f1_score)


# In[57]:


from sklearn.decomposition import PCA


# In[58]:


pca = PCA(n_components=0.99)


# In[63]:


X_train_reduced = pca.fit_transform(X_train)


# In[64]:


pca.n_components


# In[65]:


np.sum(pca.n_components)


# In[66]:


X_train_recovered = pca.inverse_transform(X_train_reduced)


# In[67]:


def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

plt.figure(figsize=(7, 4))
plt.subplot(121)
# Plotting 'original' image
plot_digits(X_train[::2100])
plt.title("Original", fontsize=16)
plt.subplot(122)
# Plotting the corresponding 'recovered' image
plot_digits(X_train_recovered[::2100])
plt.title("Compressed", fontsize=16)
plt.show()


# In[76]:


from sklearn.model_selection import GridSearchCV


# In[81]:


param_grid = [
    {
        "lr__multi_class":["multinomial"],
        "lr__solver":["lbfgs"],
        "lr__C":[5],
        "rf__n_estimators":[20],
        "rf__max_depth":[10, 15],
    }]


# In[82]:


log_clf_ens = LogisticRegression(C=10, solver="lbfgs", multi_class="multinomial", random_state=42)


# In[85]:


rnd_clf_ens = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)
from sklearn.ensemble import VotingClassifier


# In[86]:


voting_clf_grid_search = VotingClassifier(
estimators=[('lr', log_clf_ens), ('rf', rnd_clf_ens)],
voting='soft')


# In[87]:


grid_search = GridSearchCV(voting_clf_grid_search, param_grid, cv=3, scoring="neg_mean_squared_error")


# In[88]:


grid_search.fit(X_train_reduced, y_train)


# In[89]:


grid_search.best_params_


# In[90]:


grid_search.best_estimator_


# In[92]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[93]:


type(grid_search.best_params_.get("rf__max_depth"))


# In[103]:


final_model = grid_search.best_estimator_


# In[104]:


X_test_reduced = pca.transform(X_test)


# In[105]:


y_test_predict = final_model.predict(X_test_reduced)


# In[106]:


confusion_matrix(y_test, y_test_predict)


# In[107]:


final_accuracy = accuracy_score(y_test, y_test_predict)
final_precision = precision_score(y_test, y_test_predict, average='weighted')
final_recall = recall_score(y_test, y_test_predict, average='weighted')
final_f1_score = f1_score(y_test, y_test_predict, average='weighted')


# In[108]:


print("Final Accuracy: ", final_accuracy)
print("Final Precision: ", final_precision)
print("Final Recall: ", final_recall)
print("Final F1 Score: ", final_f1_score)


# In[109]:


y_test[0]

y_test_predict[0]

showImage(X_test[0])


# In[ ]:




