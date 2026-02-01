from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    model = LogisticRegression(max_iter=200)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    acc = accuracy_score(yte, preds)
    print(f"accuracy={acc:.3f}")

if __name__ == "__main__":
    main()
