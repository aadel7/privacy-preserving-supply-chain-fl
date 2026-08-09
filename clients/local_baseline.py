import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from client_node import load_data

warnings.simplefilter("ignore")

def main():
    print("Loading local data silo...")
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"Training isolated model on {len(X_train)} samples...")
    model = LogisticRegression(max_iter=1000)
    
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print("\n" + "="*40)
    print("  ISOLATED LOCAL BASELINE (SINGLE SILO)")
    print("="*40)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()