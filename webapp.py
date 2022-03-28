import numpy as np
import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

# load dataset
prop = pd.read_csv('Properties.csv')

# Replace city name with wrong names
prop['city'].replace('North', 'Isheri North', inplace=True)
prop['city'].replace('Island', 'Lagos Island', inplace=True)

def island_flag(prop):
    if (prop['city']=='Lekki') or (prop['city']=='Victoria Island') or (prop['city']=='Ajah') \
     or (prop['city']=='Ikoyi') or (prop['city']=='Eko Atlantic City') or (prop['city']=='Epe')| (prop['city']=='Lagos Island'):
        return 1
    else:
        return 0

prop['island_flag'] = prop.apply(island_flag, axis=1)

#create new features
prop['estate_flag'] = prop['location'].apply(lambda x: len([c for c in str(x).lower().split() if "estate" in c]))
prop['terrace_flag'] = prop['title'].apply(lambda x: len([c for c in str(x).lower().split() if "terraced" 
                              ""                                    in c or "terrace" in c or "detached" in c
                                                                 or "duplex" in c]))
prop['new_flag'] = prop['title'].apply(lambda x: len([c for c in str(x).lower().split()
                                                              if "new" in c or "newly" in c or "renovated" in c]))
prop['luxury_flag'] = prop['title'].apply(lambda x: len([c for c in str(x).lower().split()
                                                              if "luxury" in c or "executive" in c or "luxurious" in c]))
prop['serviced_flag'] = prop['title'].apply(lambda x: len([c for c in str(x).lower().split()
                                                                           if "serviced" in c]))
prop['studio_flag'] = prop['title'].apply(lambda x: len([c for c in str(x).lower().split()
                                                                           if "studio" in c]))
prop['miniflat_flag'] = prop['description'].apply(lambda x: len([c for c in str(x).lower().split() if "mini" in c]))
prop['selfcontain_flag'] = prop['description'].apply(lambda x: len([c for c in str(x).lower().split() if "self" in c]))

# Create new dataframe by removing features we won't be needing
prop2 = prop.drop(['page','title', 'description', 'location','garage'], axis='columns')

### Removing Outliers
prop3 = prop2[~(prop2['price']>15000000)]

prop4 = prop3[~(prop3['bedrooms']>=5)]

prop5 = prop4[~(prop4['bathrooms']>=4)]

prop6 = prop5[~(prop5['toilets']>=5)]


prop7 = prop6[~(prop6['bathrooms']>prop6['bedrooms']+1)]

prop8 = prop7[~(prop7['bathrooms']>prop7['toilets']+1)]

prop9 = prop8[~(prop8['toilets'] >prop8['bedrooms']+1)]

dummies = pd.get_dummies(prop9['city'])

prop10 = pd.concat([prop9, dummies.drop('Yaba',axis=1)], axis=1)

prop11 = prop10.drop('city',axis=1)

X = prop11.drop('price', axis=1)
y = prop11['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
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

# find_best_model(X,y)

rfr = RandomForestRegressor(criterion='mse',random_state=42)
rfr.fit(X_train, y_train)


with open('Lagos_properties_price_model.pickle','wb') as f:
    pickle.dump(rfr, f)
    
pickle_in = open('Lagos_properties_price_model.pickle', 'rb')
model = pickle.load(pickle_in)

### Let's predict prices

def predict_price(city,bedroom,bathroom,toilet,serviced=False):
    loc_index = np.where(X.columns==city)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = bedroom
    x[1] = bathroom
    x[2] = toilet
    x[7] = serviced
    if loc_index >= 0 :
        x[loc_index] = 1
        
    return '₦{:,}'.format(int(model.predict([x])[0]))

   


def  main():
    st.title('House Price Prediction')



    html_temp = """
    <div style="background-color:#f63366;
        border-radius: 25px;
        padding:5px">
    <h2 style="color:white;
        text-align:center;">House Price Prediciton ML APP</h2>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html=True)
    city = st.selectbox('Choose a Location',('Ikoyi', 'Victoria Island', 'Lekki', 'Ajah', 'Isheri North',
       'Maryland', 'Ogudu', 'Ikeja', 'Ikorodu', 'Shomolu',
       'Gbagada', 'Magodo', 'Ilupeju', 'Surulere', 'Ketu', 'Odofin',
       'Ojodu', 'Ojota', 'Kosofe', 'Agege', 'Isolo', 'Mushin', 'Ijede',
       'Ayobo', 'Ipaja', 'Ikotun', 'Eko Atlantic City', 'Alimosho', 'Ojo',
       'Ifako-Ijaiye', 'Ijaiye', 'Epe', 'Lagos Island', 'Oshodi', 'Idimu',
       'Apapa'))
    bedrooms = st.selectbox('Number of Bedrooms',(0,1,2,3,4))
    bathrooms = st.selectbox('Number of Bathrooms',(1,2,3,))
    toilets = st.selectbox('Number of Toilets',(1,2,3,4))
    
    serviced = st.radio("Do you want a Serviced Apartment? ", (True, False))
 
    if (serviced == True):
        st.success(True)
    else:
        st.success(False)
    
        
    st.write('You selected:','\n' 
             'Location:',city,'\n' 
             '\n Bedroom:',bedrooms,'\n' 
             '\n Bathroom:',bathrooms,'\n' 
             '\n Toilet:',toilets,'\n'
             '\n Serviced:',serviced)
    
    result = ""
    
    if st.button('Predict Price'):
        result = predict_price(city,bedrooms,bathrooms,toilets,serviced)
    st.success('The price is {}'. format(result))
    if st.button('About'):
        st.text('Adebo Dolapo')
        
    
if __name__=='__main__':
    main()
