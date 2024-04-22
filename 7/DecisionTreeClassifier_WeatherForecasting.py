import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('./daily_weather.csv')

data.columns
data[data.isnull().any(axis=1)].head()
del data['number']

before_rows = data.shape[0]
print(before_rows)

data = data.dropna()
after_rows = data.shape[0]
print(after_rows)
before_rows - after_rows
clean_data = data.copy()
clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm'] >24.99) *1
clean_data['high_humidity_label'].head()
y = clean_data[['high_humidity_label']].copy()
y.head()
clean_data['relative_humidity_3pm'].head()
y.head()

morning_features = ['air_pressure_9am', 'air_temp_9am', 'avg_wind_direction_9am',
       'avg_wind_speed_9am', 'max_wind_direction_9am', 'max_wind_speed_9am',
       'rain_accumulation_9am', 'rain_duration_9am', 'relative_humidity_9am']

x=clean_data[morning_features].copy()
x.columns
y.columns
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=324)
humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10,random_state=0)
humidity_classifier.fit(X_train,y_train)

y_predicted = humidity_classifier.predict(X_test)
y_predicted[:10]
y_test['high_humidity_label'][:10]
accuracy_score(y_test,y_predicted)*100
