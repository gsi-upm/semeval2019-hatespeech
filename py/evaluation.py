from sklearn import svm, linear_model, tree, ensemble, naive_bayes, neighbors, gaussian_process 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, vstack, coo_matrix
from sklearn import metrics
from tqdm import tqdm

def createModel(clf):
    
    model = Pipeline([
        #('feature_selection', SelectKBest(chi2, k=300)),
        #('to_dense', DenseTransformer()),
        ('clf', clf)
    ])
    return model

def evaluate(features_train, y_train, features_test, y_test, classifiers, verbose=False, n_jobs=1):

    results = {}    
    try: # in case features are not sparse
        features = vstack([features_train, features_test])
    except ValueError:
        features = np.concatenate((features_train, features_test))
        
    # Cross-validation
    #cv = KFold(n_splits=3, shuffle=True, random_state=33)
    
    # Specify train and tests sets to gridsearch
    n_samples_train = features_train.shape[0]
    n_samples_test = features_test.shape[0]
    total_samples = features.shape[0]
    assert total_samples == n_samples_train+n_samples_test
    train_index = np.arange(0, n_samples_train)
    test_index = np.arange(n_samples_train, total_samples)
    cv = [[train_index, test_index]]

    y = np.concatenate((y_train, y_test), axis=None)
    # Optimization
    for nombre, modelo in tqdm(classifiers.items()):
        results[nombre] = {}
        
        model = createModel(modelo['model'])
        gs = GridSearchCV(model, modelo['params'], scoring='f1_macro', cv=cv, n_jobs=n_jobs)
        gs.fit(features, y)

        # summarize the results of the grid search
        print(nombre)
        print("Best score: ", gs.best_score_)
        print("Best params: ", gs.best_params_)
        print('\n')
        params = model.get_params()
        for param in modelo['params'].keys():
            params.update({param: gs.best_params_[param]})
        
        model.set_params(**params)
        results[nombre]['params'] = params
        
        # Training
        model.fit(features_train, y_train)
        predicted = model.predict(features_test)

        # Evaluation
        if verbose:
            print(nombre)
            
        accuracy_score = metrics.accuracy_score(y_test, predicted)
        precision_score = metrics.precision_score(y_test, predicted, average='macro')
        recall_score = metrics.recall_score(y_test, predicted, average='macro')
        f1_score = metrics.f1_score(y_test, predicted, average='macro')
    
        results[nombre]['accuracy'] = accuracy_score
        results[nombre]['precision'] = precision_score
        results[nombre]['recall'] = recall_score
        results[nombre]['f1'] = f1_score
        results[nombre]['predicted'] = predicted
        
        if verbose:
            print("Accuracy:", accuracy_score)
            print("Precision:", precision_score)
            print("Recall:", recall_score)
            print("F1_score:", f1_score)
            print('\n')
        
    return results
