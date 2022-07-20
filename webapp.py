import numpy as np
import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
# load dataset
prop = pd.read_csv('Properties.csv')

# Renaming some cities properly to aid our prediciton perform better
prop['city'].replace('North', 'Isheri North', inplace=True)
prop['city'].replace('Island', 'Lagos Island', inplace=True)

def island_flag(prop):
    if (prop['city']=='Lekki') or (prop['city']=='Victoria Island') or (prop['city']=='Ajah') \
     or (prop['city']=='Ikoyi') or (prop['city']=='Eko Atlantic City') \
    or (prop['city']=='Epe') or (prop['city']=='Lagos Island'):
        return 1
    else:
        return 0

# Creating Island features
prop['island_flag'] = prop.apply(island_flag, axis=1)
# Creating Estate feature
prop['estate_flag'] = prop['location'].apply(lambda x: len([c for c in str(x).lower().split() if "estate" in c]))
# Creating terrace/duplex/detached houses feature
prop['terrace_flag'] = prop['title'].apply(lambda x: len([c for c in str(x).lower().split() if "terraced" 
                                                          in c or "terrace" in c or "detached" in c
                                                                 or "duplex" in c]))
# Creating new/renovated houses feature
prop['new_flag'] = prop['title'].apply(lambda x: len([c for c in str(x).lower().split()
                                                              if "new" in c or "newly" in c or "renovated" in c]))
# Creating luxurious/executive/exquisite houses feature
prop['luxury_flag'] = prop['title'].apply(lambda x: len([c for c in str(x).lower().split()
                                                              if "luxury" in c or "executive" in c or "luxurious" in c
                                                        or "exquisite"in c or "excellent" in c]))
# Creating Serviced houses feature
prop['serviced_flag'] = prop['title'].apply(lambda x: len([c for c in str(x).lower().split()
                                                                           if "serviced" in c]))
# Creating studio apartment feature
prop['studio_flag'] = prop['title'].apply(lambda x: len([c for c in str(x).lower().split()
                                                                           if "studio" in c]))
# Creating big/spacious houses feature
prop['big_flag'] = prop['title'].apply(lambda x: len([c for c in str(x).lower().split() if "spacious" in c
                                                     or "big" in c]))
# Creating self-contain houses feature
prop['selfcontain_flag'] = prop['description'].apply(lambda x: len([c for c in str(x).lower().split() if "self" in c]))
# Dropping irrelevant features
prop2 = prop.drop(['page','title', 'description', 'location','garage'], axis='columns')
# Removing prices higher than 15,000,000
prop3 = prop2[~(prop2['price']>15000000)]
# Removing bedrooms greater than 5
prop4 = prop3[~(prop3['bedrooms']>5)]
# Removing bathrooms greater than 5
prop5 = prop4[~(prop4['bathrooms']>5)]
# Removing toilets greater than 5
prop6 = prop5[~(prop5['toilets']>5)]

# Also, we check if number of bathrooms is greater than bedrooms and also if number of toilets is less than bathrooms
prop7 = prop6[~(prop6['bathrooms']>prop6['bedrooms']+2)]
prop8 = prop7[~(prop7['bathrooms']>prop7['toilets']+2)]
prop9 = prop8[~(prop8['toilets'] >prop8['bedrooms']+2)]


dummies = pd.get_dummies(prop9['city'])
prop10 = pd.concat([prop9, dummies], axis=1).drop('city',axis=1)

X = prop10.drop('price', axis=1)
y = prop10['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

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
    x[8] = serviced
    if loc_index >= 0 :
        x[loc_index] = 1
        
    return '₦{:,}'.format(int(model.predict([x])[0]))


def  main():
    st.title('House Price Prediction Project')



    html_temp = """
    <div style="background-color:#f63366;
        border-radius: 25px;
        padding:5px">
    <h2 style="color:white;
        text-align:center;">Lagos House Prices Prediciton ML APP</h2>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html=True)
    city = st.selectbox('Choose a Location',('Lekki','Ikoyi', 'Victoria Island', 'Ajah', 'Isheri North',
       'Maryland', 'Ogudu', 'Ikeja', 'Ikorodu', 'Shomolu','Yaba',
       'Gbagada', 'Magodo', 'Ilupeju', 'Surulere', 'Ketu', 'Odofin',
       'Ojodu', 'Ojota', 'Kosofe', 'Agege', 'Isolo', 'Mushin', 'Ijede',
       'Ayobo', 'Ipaja', 'Ikotun', 'Eko Atlantic City', 'Alimosho', 'Ojo',
       'Ifako-Ijaiye', 'Ijaiye', 'Epe', 'Lagos Island', 'Oshodi', 'Idimu',
       'Apapa'))
    bedrooms = st.selectbox('Number of Bedrooms',(0,1,2,3,4,5))
    bathrooms = st.selectbox('Number of Bathrooms',(1,2,3,4,5))
    toilets = st.selectbox('Number of Toilets',(1,2,3,4,5))
    
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
