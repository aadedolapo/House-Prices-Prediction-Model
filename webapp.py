import numpy as np
import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# load dataset
prop = pd.read_csv('Properties.csv')

#create new features
prop['estate_flag'] = prop['location'].apply(lambda x: len([c for c in str(x).lower().split() if "estate" in c]))
prop['terrace_flag'] = prop['title'].apply(lambda x: len([c for c in str(x).lower().split() if "terraced" 
                                                                  in c or "terrace" in c or "detached" in c
                                                                 or "duplex" in c]))
prop['luxury_flag'] = prop['title'].apply(lambda x: len([c for c in str(x).lower().split()
                                                              if "new" in c or "luxury" in c or "newly" in c
                                                              or "executive" in c or "luxurious" in c
                                                                        or "renovated" in c]))
prop['serviced_flag'] = prop['title'].apply(lambda x: len([c for c in str(x).lower().split()
                                                                           if "serviced" in c or "studio" in c]))
prop['miniflat_flag'] = prop['description'].apply(lambda x: len([c for c in str(x).lower().split() if "mini" in c]))
prop['selfcontain_flag'] = prop['description'].apply(lambda x: len([c for c in str(x).lower().split() if "self" in c]))

prop2 = prop.drop(['page', 'title', 'garage', 'description', 'location'], axis='columns')

### These Prices are too high for rent. These are outliers

prop3 = prop2[~(prop2['price']>20000000)]

### Also, we check if number of bathrooms is greater than bedrooms and also if number of toilets is less than bathrooms

prop4 = prop3[~(prop3['bathrooms']>prop3['bedrooms']+1)]

prop5 = prop3[~(prop3['bathrooms']>prop3['toilets']+1)]

dummies = pd.get_dummies(prop5['city'])

prop6 = pd.concat([prop5, dummies.drop('Yaba',axis=1)], axis=1)

prop7 = prop6.drop('city',axis=1)

X = prop7.drop('price', axis=1)
y = prop7['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

# ShuffleSplit will randomize the sample so each of the fold have eqaul distribution of my data samples

cross_val_score(lr, X, y, cv=cv)

## Hyperparamater tunning

### Using Grid Search CV to find the best model

def find_best_model(X,y):
    models = {
        'linear_regression':{
            'model':LinearRegression(),
            'params':{'normalize':[True, False]
            }
        },
        'random_forest':{
            'model':RandomForestRegressor(),
            'params':{
                'criterion' :['mse']
                }
        },
        'lasso':{
            'model':Lasso(),
            'params':{
                'alpha':[1,2],
                'selection':['random','cyclic']
            }
        },
        'decision_tree':{
            'model':DecisionTreeRegressor(),
            'params':{
                'criterion':['mse','friedman_mse'],
                'splitter':['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for model_name, config in models.items():
        gridsearch = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gridsearch.fit(X,y)
        scores.append({
            'model': model_name,
            'best_score':gridsearch.best_score_,
            'best_params':gridsearch.best_params_
        })
        
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])
#find_best_model(X,y)

rfr = RandomForestRegressor(criterion='mse',random_state=42)
rfr.fit(X_train, y_train)


with open('House_price_model.pickle','wb') as f:
    pickle.dump(rfr, f)
    
pickle_in = open('House_price_model.pickle', 'rb')
model = pickle.load(pickle_in)

#def predict_price(city,bedrooms,bathrooms,toilets,garage):
def predict_price(city,bedrooms,bathrooms,toilets):
    loc_index = np.where(X.columns==city)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = bedrooms
    x[1] = bathrooms
    x[2] = toilets
    if loc_index >= 0 :
       x[loc_index] = 1
        
    return '₦{:,}'.format(int(model.predict([x])[0]))
   


def  main():
    html_temp = """
    <div style="background-color:#f63366;
        border-radius: 25px;
        padding:5px">
    <h2 style="color:white;
        text-align:center;">House Price Prediciton ML APP</h2>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html=True)
    city = st.selectbox('Choose a Location',('Agege', 'Ajah', 'Alimosho', 'Apapa', 'Ayobo',
       'Eko Atlantic City', 'Epe', 'Gbagada', 'Idimu', 'Ifako-Ijaiye',
       'Ijaiye', 'Ijede', 'Ikeja', 'Ikorodu', 'Ikotun', 'Ikoyi', 'Ilupeju',
       'Ipaja', 'Island', 'Isolo', 'Ketu', 'Kosofe', 'Lekki', 'Magodo',
       'Maryland', 'Mushin', 'North', 'Odofin', 'Ogudu', 'Ojo', 'Ojodu',
       'Ojota', 'Oshodi', 'Shomolu', 'Surulere', 'Victoria Island'))
    bedrooms = st.selectbox('Number of Bedrooms',(0,1,2,3,4,5,6,7,8))
    bathrooms = st.selectbox('Number of Bathrooms',(1,2,3,4,5,6,7,8))
    toilets = st.selectbox('Number of Toilets',(1,2,3,4,5,6,7,8,9))
        
    st.write('You selected:','\n' 
             'Location:',city,'\n' 
             '\n Bedroom:',bedrooms,'\n' 
             '\n Bathroom:',bathrooms,'\n' 
             '\n Toilet:',toilets)
    
    result = ""
    
    if st.button('Predict Price'):
        result = predict_price(city,bedrooms,bathrooms,toilets)
    st.success('The price is {}'. format(result))
    if st.button('About'):
        st.text('Adebo Dolapo')
        
    
if __name__=='__main__':
    main()
