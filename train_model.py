from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def visualize_model(grid_search):

    print("Best Parameters:", grid_search.best_params_)
    results = pd.DataFrame(grid_search.cv_results_)
    cols = [col for col in results.columns if col.startswith("param_")] + ["mean_train_score", "mean_test_score"]
    results = results[cols]
    print(results)
    results['gap'] = results['mean_train_score'] - results['mean_test_score']
    results.sort_values(by='gap', ascending=False)

    plt.plot(results['mean_train_score'], label='Train Score')
    plt.plot(results['mean_test_score'], label='Validation Score')
    plt.legend()
    plt.title('Train vs Validation Score in GridSearchCV')
    plt.show()
 
def best_parameter(parametr,model, X_train, y_train):
    grid_search = GridSearchCV(model, parametr, cv=5, n_jobs=-1, scoring='accuracy',return_train_score=True)
    grid_search.fit(X_train, y_train)

    visualize_model(grid_search)

    calibrated_clf = CalibratedClassifierCV(grid_search.best_estimator_, method='sigmoid', cv=5 )
    calibrated_clf.fit(X_train, y_train)

    return grid_search, calibrated_clf 



def predict_model(model, X,mode="predict"): 
    if mode == "proba":
        return model.predict_proba(X)
    else:
        return model.predict(X)