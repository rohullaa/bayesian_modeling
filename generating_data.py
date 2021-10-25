import numpy as np
import pandas as pd

def generate_binary_data(num_features, N, correlation=[0.9, 0.5]):
    data = np.random.choice(2, size=(N, num_features))
    df = pd.DataFrame(data)
    df["Target"] = np.zeros(N).astype(int)
    for i, cor in enumerate(correlation):
        if i >= num_features:
            break

        df["Target"] |= df.iloc[:, i] * np.random.choice(2, size=N, p=[(1-cor), cor])

    return df.iloc[:, :num_features], df["Target"]


X,y = generate_binary_data(10,100_000, correlation=[0.9, 0.5])
