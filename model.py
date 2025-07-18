import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
df = pd.read_csv("homeprices_banglore.csv")
from sklearn.model_selection import train_test_split
X = df.drop('price',axis='columns')
y=df['price'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)

pickle.dump(model,open('model.pkl','wb'))