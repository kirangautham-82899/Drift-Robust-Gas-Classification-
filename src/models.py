"""The baseline models of the project, all with scikit-learn **default** settings.

No hyperparameter is tuned in the preliminary phase; only ``random_state`` is fixed (42)
so that results are reproducible. ``Majority class`` is not one of the seven compared
models: it always predicts the most frequent training gas and gives the "no skill" floor.
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42


def default_models():
    """Return ``{name: estimator}`` for the seven compared models (default settings)."""
    return {
        "kNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(random_state=SEED),
        "Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }


def reference_model():
    """The no-skill reference (always predicts the most frequent training gas)."""
    return {"Majority class": DummyClassifier(strategy="most_frequent")}
