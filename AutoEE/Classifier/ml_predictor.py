import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def single_lgbm(train_data,train_lable):
    print('----------LightGBM------------')
    import lightgbm as lgb
    train_data = train_data.cpu().numpy()
    train_lable = train_lable.cpu().numpy()
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_lable, test_size=0.5, random_state=42)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {  
        'boosting_type': 'goss',  
        'objective': 'binary', 
        'metric': 'binary_error',  
        'num_leaves': 31,  
        'min_data_in_leaf': 20,  
        'learning_rate': 0.1,  
        'feature_fraction': 0.8,  
        'bagging_fraction': 0.8,  
        'bagging_freq': 0,  
        'lambda_l1': 0.1,  
        'lambda_l2': 0.05,  
        'min_gain_to_split': 0.2,  
        'verbose': -1, 
        'early_stopping_rounds':5
    }
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=lgb_eval)
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    binary_pred = [1 if i > 0.5 else 0 for i in y_pred]
    accuracy = sum([1 if i == j else 0 for i, j in zip(binary_pred, y_test)]) / len(y_test)
    acc = sum([1 if i == 1 and j == 0 else 0 for i, j in zip(binary_pred, y_test)])
    acc_err = sum([1 if i == 1 and j == 1 else 0 for i, j in zip(binary_pred, y_test)])
    print('binary acc:', accuracy,acc,acc_err)
    return accuracy,gbm


    
def SVC_binary(train_data,train_lable):
    print('----------------SVC----------------')
    from sklearn.svm import SVC

    train_data = train_data.cpu().numpy()
    train_lable = train_lable.cpu().numpy()
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_lable, test_size=0.8, random_state=42)
    
    svc = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None)
    
    svc.fit(X_train,y_train)
    
    y_pred = svc.predict(X_test)
    print('SVC Accuracy:' ,accuracy_score(y_test,y_pred))
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['linear','rbf', 'poly', 'sigmoid'],
        'degree': [2, 3, 4],
    }
    
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    print('Best Parameters:', grid_search.best_params_)
    print('Best Score:', grid_search.best_score_)
    best_svc = grid_search.best_estimator_
    y_pred = best_svc.predict(X_test)
    print('Accuracy with best parameters:', accuracy_score(y_test, y_pred))
    
def KNN_binary(train_data,train_lable):
    print('-------KNN-------')
    
    train_data = train_data.cpu().numpy()
    train_lable = train_lable.cpu().numpy()
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_lable, test_size=0.8, random_state=42)
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', p=2)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    print('KNN Accuracy:', accuracy_score(y_test, y_pred))
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]
    }
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    
    print('Best Parameters:', grid_search.best_params_)
    print('Best Score:', grid_search.best_score_)

    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)
    
    print('Accuracy with best parameters:', accuracy_score(y_test, y_pred))

def DecisionTree(train_data,train_lable):
    print('-----------Decision Tree-------------')
    from sklearn.tree import DecisionTreeClassifier
    train_data = train_data.cpu().numpy()
    train_lable = train_lable.cpu().numpy()
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_lable, test_size=0.8, random_state=42)
    
    dtc = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, 
                             min_samples_split=2, min_samples_leaf=1, random_state=42)
    
    dtc.fit(X_train, y_train)
    
    y_pred = dtc.predict(X_test)
    
    print('DTC Accuracy:', accuracy_score(y_test, y_pred))
    
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10]
    }
    
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', verbose=1)
    
    grid_search.fit(X_train, y_train)
    
    print('Best Parameters:', grid_search.best_params_)
    print('Best Score:', grid_search.best_score_)
    
    best_dtc = grid_search.best_estimator_
    y_pred = best_dtc.predict(X_test)
    print('Accuracy with best parameters:', accuracy_score(y_test, y_pred))
    

    


def layer_classify(train_data,train_lable):
    layer = []
    import lightgbm as lgb
    for i in range(32):
        train_data_layer = train_data[i::32]
        train_data_layer = train_data_layer[:,:-1]
        train_lable_layer = train_lable[i::32]
        print('-----------Layer:',i,'-----------')
        # print('data.shape',train_data_layer.shape)
        # print('label.shape',train_lable_layer.shape)
        acc,model = single_lgbm(train_data_layer,train_lable_layer)
        model.save_model('../results/model'+str(i)+'.txt')
        # acc = KNN_binary(train_data_layer,train_lable_layer)
        # acc = DecisionTree(train_data_layer,train_lable_layer)
    #     if acc > 0.6:
    #         layer.append(i)
    # print(layer)
        
if __name__ == '__main__':

    train_data = torch.load('/home/xujiaming/xujiaming/research/ASPLOS-24/results/feature.pt')
    train_label = torch.load('/home/xujiaming/xujiaming/research/ASPLOS-24/results/train_label.pt')
    print('-------------Single--------------')
    acc,model = single_lgbm(train_data,train_label)
    model.save_model('../results/model.txt')
    # print("-------------Layer--------------")
    # layer_classify(train_data,train_label)
    # SVC_binary(train_data,train_label)
    # KNN_binary(train_data,train_label)
    # DecisionTree(train_data,train_label)
    # layer_classify(train_data,train_label)