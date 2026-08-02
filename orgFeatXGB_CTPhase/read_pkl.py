import pickle

with open('xgb_vindr.pkl', 'rb') as f:
    data = pickle.load(f)

print(type(data))
print(data.keys() if isinstance(data, dict) else data)
