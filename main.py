
#importing packages 
import json
import pandas as pd
import numpy as np
import hashlib


#load iris dataset
iris = pd.read_csv('iris.csv')
iris.head()


#load json file
with open('sample.json') as json_file:
    data = json.load(json_file)


#Select target variable and prediction type
target = data["design_state_data"]["target"]["target"]
prediction_type = data["design_state_data"]["target"]["prediction_type"]
X = iris.drop(target, axis=1)
y = iris[target]

#Feature Handling
feature_handling = data["design_state_data"]['feature_handling']

#Tokenizing and hashing
def tokenize_and_hash(text):
    # Tokenize the text
    tokens = text.lower().split()
    # Hash each token using SHA-256 algorithm
    hashed_tokens = [hashlib.sha256(token.encode()).hexdigest() for token in tokens]
    # Join the hashed tokens into a single string
    hashed_text = ''.join(hashed_tokens)
    return hashed_text


for i in feature_handling:
    x=feature_handling[i]
    if x['feature_variable_type']== 'numerical':
        if x['feature_details']['impute_with']== 'custom':
            custom = float(x['feature_details']['impute_value'])
            iris[i].fillna(custom, inplace=True)
        elif x['feature_details']['impute_with']== 'Average of values':
            m = iris[i].mean()
            iris[i].fillna(m, inplace=True)
    elif x['feature_variable_type']== 'text':
        if x['feature_details']['text_handling']== 'Tokenize and hash':            
            n=int(x['feature_details']['hash_columns'])
            for j in range(1, n+1):
                var_name = f"hash_columns{i}"  # create the variable name
                iris[var_name] = iris[i].apply(tokenize_and_hash)


#Feature reduction
feature_reduction = data["design_state_data"]['feature_reduction']

#Based on screenshot
'''
for i in feature_reduction:
    x=feature_reduction[i]
    if i["is_selected"]== true:
        if i == "Principal Componenet Analysis":
            num_of_features_to_keep = x["num_of_features_to_keep"]
            pca = PCA(n_components=num_of_features_to_keep)
            X = pca.fit_transform(X)
        if i == "Correlation with target":
            num_of_features_to_keep = x["num_of_features_to_keep"]
            corr_matrix = data.corr()
            important_features = corr_matrix['target'].abs().sort_values(ascending=False)[:num_of_features_to_keep].index
            X = X[important_features]
        if i == "Tree-Based":
            n_features = int(x["num_of_features_to_keep"])
            n_estimators = int(x["num_of_trees"])
            max_depth = int(x['depth_of_trees'])
            clf = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
            clf.fit(X, y)
            importances = clf.feature_importances_
            indices = importances.argsort()[::-1]
            top_k_features = X.columns[indices][:n_features]
            X = X[top_k_features]
'''

#Based on json file

if feature_reduction["feature_reduction_method"] == "Tree-Based":
    n_features = int(feature_reduction["num_of_features_to_keep"])
    n_estimators = int(feature_reduction["num_of_trees"])
    max_depth = int(feature_reduction['depth_of_trees'])
    clf = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
    clf.fit(X, y)
    importances = clf.feature_importances_
    indices = importances.argsort()[::-1]
    top_k_features = X.columns[indices][:n_features]
    X = X[top_k_features]

#Spliting data
from sklearn.model_selection import train_test_split
X=X.drop('species', axis = 1)
test_s=data["design_state_data"]["train"]["train_ratio"]
if test_s !=0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s)

#Model and Parameter Selection

alg = data["design_state_data"]["algorithms"]

model={}
if prediction_type == "Regression":
    for i in alg:
        if alg[i]["is_selected"]==True :
            if alg[i]["model_name"]== 'Random Forest Regressor' :
                from sklearn.ensemble import RandomForestRegressor
                min_trees = int(alg[i]["min_trees"])
                max_trees = int(alg[i]["max_trees"])
                min_depth = int(alg[i]["min_depth"])
                max_depth = int(alg[i]["max_depth"])
                if alg[i]["feature_sampling_statergy"] =='Default':
                    feature_sampling_strategy = None
                else :
                    feature_sampling_strategy = alg[i]["feature_sampling_statergy"]
                min_samples_per_leaf_min_value = int(alg[i]["min_samples_per_leaf_min_value"])
                min_samples_per_leaf_max_value = int(alg[i]["min_samples_per_leaf_max_value"])
                parallelism = int(alg[i]["parallelism"])            
                rf = RandomForestRegressor()                                        
                rf.fit(X_train, y_train)
                score = rf.score(X_test, y_test)
                model["random_forest"]={
                'model': rf,
                'params' : {
                'n_estimators': list(range(min_trees,max_trees+1)),
                'max_depth': list(range(min_depth,max_depth+1)),
                'max_features': [feature_sampling_strategy],
                'min_samples_leaf': list(range(min_samples_per_leaf_min_value,min_samples_per_leaf_max_value))
                    }
                }
                
            if alg[i]["model_name"]== 'Gradient Boosted Trees' :
                from sklearn.ensemble import GradientBoostingRegressor
                num_of_BoostingStages = alg[i]['num_of_BoostingStages']               
                min_depth = int(alg[i]["min_depth"])
                max_depth = int(alg[i]["max_depth"])
                feature_sampling_strategy = alg[i]["feature_sampling_statergy"]
                use_deviance = alg[i]['use_deviance']
                use_exponential = alg[i]['use_exponential']
                fixed_number = alg[i]['fixed_number']
                min_subsample = alg[i]['min_subsample']
                max_subsample = alg[i]['max_subsample']
                min_stepsize = alg[i]['min_stepsize']
                max_stepsize = alg[i]['max_stepsize']
                min_iter = alg[i]['min_iter']
                max_iter = alg[i]['max_iter'] 
              
                rf = GradientBoostingRegressor()
                rf.fit(X_train, y_train)
                score = rf.score(X_test, y_test)
                
                model["gradient_boost"]={
                'model': rf,
                'params' : {
                'n_estimators': num_of_BoostingStages,
                'max_depth': list(range(min_depth,max_depth+1)),
                'learning_rate': list(np.arange(min_stepsize,max_stepsize+0.1,0.1)),
                'subsample':list(range(min_subsample,max_subsample+1))
                    }
                }
                
            if alg[i]["model_name"]== 'LinearRegression' :
                from sklearn.linear_model import LinearRegression
                parallelism = int(alg[i]["parallelism"])               
                min_regparam = alg[i]['min_regparam']
                max_regparam = alg[i]['max_regparam']
                min_elasticnet = alg[i]['min_elasticnet']
                max_elasticnet = alg[i]['max_elasticnet']
                min_iter = alg[i]['min_iter']
                max_iter = alg[i]['max_iter'] 
                results = []   
                rf = LinearRegression()                         
                rf.fit(X_train, y_train)
                score = rf.score(X_test, y_test)
                
                model["linear_regression"]={
                'model': rf,
                'params' : {
                    }
                }
                
            if alg[i]["model_name"]== 'RidgeRegression' :
                from sklearn.linear_model import Ridge
                parallelism = int(alg[i]["parallelism"])               
                min_regparam = alg[i]['min_regparam']
                max_regparam = alg[i]['max_regparam']
                min_elasticnet = alg[i]['min_elasticnet']
                max_elasticnet = alg[i]['max_elasticnet']
                min_iter = alg[i]['min_iter']
                max_iter = alg[i]['max_iter'] 
                rf = Ridge()                         
                rf.fit(X_train, y_train)
                score = rf.score(X_test, y_test)
                
                model["ridge_regression"]={
                'model': rf,
                'params' : {
                    'max_iter': list(range(min_iter,max_iter+1))
                    
                    }
                }
                                
            if alg[i]["model_name"]== 'Lasso Regression' :
                from sklearn.linear_model import Lasso
                parallelism = int(alg[i]["parallelism"])               
                min_regparam = alg[i]['min_regparam']
                max_regparam = alg[i]['max_regparam']
                min_elasticnet = alg[i]['min_elasticnet']
                max_elasticnet = alg[i]['max_elasticnet']
                min_iter = alg[i]['min_iter']
                max_iter = alg[i]['max_iter']   
                rf = Lasso()                         
                rf.fit(X_train, y_train)
                score = rf.score(X_test, y_test)
                
                model["lasso_regression"]={
                'model': rf,
                'params' : {
                    'max_iter': list(range(min_iter,max_iter+1))
                    
                    }
                }
            

                                
            if alg[i]["model_name"]== 'Decision Tree' :
                from sklearn.tree import DecisionTreeRegressor
                min_depth = int(alg[i]["min_depth"])
                max_depth = int(alg[i]["max_depth"])
                min_samples_per_leaf = alg[i]['min_samples_per_leaf']
                
                 
                rf = DecisionTreeRegressor()                         
                rf.fit(X_train, y_train)
                score = rf.score(X_test, y_test)
                model["decision_tree"]={
                'model': rf,
                'params' : {
                    'max_depth': list(range(min_depth,max_depth+1)),
                    'min_samples_leaf':min_samples_per_leaf
                    
                    }
                }
                
                
                
                
                
if prediction_type == "Classification":
    for i in alg:
        if alg[i]["is_selected"]==True :
            if alg[i]["model_name"]== 'Random Forest Classifier' :
                from sklearn.ensemble import RandomForestClassifier
                min_trees = int(alg[i]["min_trees"])
                max_trees = int(alg[i]["max_trees"])
                min_depth = int(alg[i]["min_depth"])
                max_depth = int(alg[i]["max_depth"])
                if alg[i]["feature_sampling_statergy"] =='Default':
                    feature_sampling_strategy = None
                else :
                    feature_sampling_strategy = alg[i]["feature_sampling_statergy"]
                min_samples_per_leaf_min_value = int(alg[i]["min_samples_per_leaf_min_value"])
                min_samples_per_leaf_max_value = int(alg[i]["min_samples_per_leaf_max_value"])
                parallelism = int(alg[i]["parallelism"])            
                rf = RandomForestClassifier()                                       
                rf.fit(X_train, y_train)
                score = rf.score(X_test, y_test)
                model["random_forest"]={
                'model': rf,
                'params' : {
                'n_estimators': list(range(min_trees,max_trees+1)),
                'max_depth': list(range(min_depth,max_depth+1)),
                'max_features': [feature_sampling_strategy],
                'min_samples_leaf': list(range(min_samples_per_leaf_min_value,min_samples_per_leaf_max_value))
                    }
                }
                
            if alg[i]["model_name"]== 'Gradient Boosted Trees' :
                from sklearn.ensemble import GradientBoostingClassifier
                num_of_BoostingStages = alg[i]['num_of_BoostingStages']               
                min_depth = int(alg[i]["min_depth"])
                max_depth = int(alg[i]["max_depth"])
                feature_sampling_strategy = alg[i]["feature_sampling_statergy"]
                use_deviance = alg[i]['use_deviance']
                use_exponential = alg[i]['use_exponential']
                fixed_number = alg[i]['fixed_number']
                min_subsample = alg[i]['min_subsample']
                max_subsample = alg[i]['max_subsample']
                min_stepsize = alg[i]['min_stepsize']
                max_stepsize = alg[i]['max_stepsize']
                min_iter = alg[i]['min_iter']
                max_iter = alg[i]['max_iter'] 
              
                rf = GradientBoostingClassifier()
                rf.fit(X_train, y_train)
                score = rf.score(X_test, y_test)
                
                model["gradient_boost"]={
                'model': rf,
                'params' : {
                'n_estimators': num_of_BoostingStages,
                'max_depth': list(range(min_depth,max_depth+1)),
                'learning_rate': list(np.arange(min_stepsize,max_stepsize+0.1,0.1)),
                'subsample':list(range(min_subsample,max_subsample+1))
                    }
                }
                
                                
            if alg[i]["model_name"]== 'Decision Tree' :
                from sklearn.tree import DecisionTreeClassifier
                min_depth = int(alg[i]["min_depth"])
                max_depth = int(alg[i]["max_depth"])
                min_samples_per_leaf = alg[i]['min_samples_per_leaf']
                
                 
                rf = DecisionTreeClassifier()                         
                rf.fit(X_train, y_train)
                score = rf.score(X_test, y_test)
                model["decision_tree"]={
                'model': rf,
                'params' : {
                    'max_depth': list(range(min_depth,max_depth+1)),
                    'min_samples_leaf':min_samples_per_leaf
                    
                    }
                }
                
            if alg[i]["model_name"]== 'Support Vector Machine' :
                from sklearn.svm import SVC
                c_value = alg[i]["c_value"]
                max_iterations = int(alg[i]["max_iterations"])
                min_samples_per_leaf = alg[i]['min_samples_per_leaf']                 
                rf = SVC()                         
                rf.fit(X_train, y_train)
                score = rf.score(X_test, y_test)
                model["decision_tree"]={
                'model': rf,
                'params' : {
                    'C': c_value
         
                    }
                }
                
            if alg[i]["model_name"]== 'Stochastic Gradient Descent' :
                from sklearn.linear_model import SGDClassifier
                alpha_value = alg[i]["alpha_value"]                 
                rf = SGDClassifier()                         
                rf.fit(X_train, y_train)
                score = rf.score(X_test, y_test)
                model["decision_tree"]={
                'model': rf,
                'params' : {
                    'alpha': alpha_value
         
                    }
                }
                
            if alg[i]["model_name"]== 'KNN' :
                from sklearn.neighbors import KNeighborsClassifier
                k_value = alg[i]["k_value"]                 
                rf = KNeighborsClassifier()                         
                rf.fit(X_train, y_train)
                score = rf.score(X_test, y_test)
                model["decision_tree"]={
                'model': rf,
                'params' : {
                    'n_neighbors': k_value
         
                    }
                }

#Hyperparameters

hyperparameters = data["design_state_data"]['hyperparameters']
c=hyperparameters['num_of_folds']

#GridSearchCV

from sklearn.model_selection import GridSearchCV
scores = []
for model_name, mp in model.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=c, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
res = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(res)



