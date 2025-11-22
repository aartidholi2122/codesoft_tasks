import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("imdb.csv", encoding='latin1')

print("Available Columns:")
print(df.columns)

# Automatically detect possible columns
possible_genre = [c for c in df.columns if 'genre' in c.lower()]
possible_director = [c for c in df.columns if 'director' in c.lower()]
possible_actor = [c for c in df.columns if 'actor' in c.lower()]
possible_rating = [c for c in df.columns if 'rating' in c.lower() or 'score' in c.lower()]

genre_col = possible_genre[0] if possible_genre else None
director_col = possible_director[0] if possible_director else None
actor_col = possible_actor[0] if possible_actor else None
rating_col = possible_rating[0] if possible_rating else None

print("\nDetected:")
print("Genre:", genre_col)
print("Director:", director_col)
print("Actors:", actor_col)
print("Rating:", rating_col)

if None in [genre_col, director_col, actor_col, rating_col]:
    print("\nError: Required columns not found!")
    exit()

df = df[[genre_col, director_col, actor_col, rating_col]]
df.dropna(inplace=True)

le = LabelEncoder()
df[genre_col] = le.fit_transform(df[genre_col])
df[director_col] = le.fit_transform(df[director_col])
df[actor_col] = le.fit_transform(df[actor_col])

X = df[[genre_col, director_col, actor_col]]
y = df[rating_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

rmse = mean_squared_error(y_test, pred) ** 0.5
print("\nRMSE:", rmse)