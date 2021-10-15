# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score


from interpret.blackbox import LimeTabular
from interpret import show


# %%
df = pd.read_csv("./data/CEE_DATA.csv", quotechar="'")

# %%
from sklearn.preprocessing import OrdinalEncoder

ode1 = OrdinalEncoder(categories=[["Average", "Good", "Vg", "Excellent"]])
df["Performance"] = ode1.fit_transform(df[["Performance"]])


# %%
X = df[
    [
        "Gender",
        "Caste",
        "coaching",
        "time",
        "Class_ten_education",
        "twelve_education",
        "medium",
        "Class_X_Percentage",
        "Class_XII_Percentage",
        "Father_occupation",
        "Mother_occupation",
    ]
]
Y = df["Performance"]


# %%
X = pd.get_dummies(X)


# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
print("Train Size Instances: ", X_train.shape[0])
print("Test Size Instances:", X_test.shape[0])


# %%
clf = svm.SVC(probability=True)
# clf = naive_bayes.MultinomialNB()
# clf = KNeighborsClassifier(n_neighbors=5)
# clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X_train, Y_train)


# %%
Y_pred = clf.predict(X_test)
print("Accuracy: ", accuracy_score(Y_test, Y_pred))
print(f"F1 Score {f1_score(Y_test, Y_pred, average='macro')}")


# %%
lime = LimeTabular(
    predict_fn=clf.predict_proba,
    data=X_train,
    explain_kwargs={"top_labels": 2},
    class_names=["Average", "Good", "Vg", "Excellent"],
)

# %%
lime_local = lime.explain_local(X_test[:2])
show(lime_local)

# %%
