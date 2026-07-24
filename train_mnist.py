"""Train a 3-layer MLP on real MNIST using only the from-scratch engine in nn/."""

import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from nn.engine import Tensor
from nn.layers import Linear, ReLU, Sequential, cross_entropy_loss
from nn.optim import SGD

SEED = 0
BATCH_SIZE = 128
EPOCHS = 15
LR = 0.1
MOMENTUM = 0.9


def load_mnist():
    print("Loading MNIST (downloads + caches to ~/scikit_learn_data on first run)...")
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    X = X.astype(np.float64) / 255.0
    y = y.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=SEED, stratify=y
    )
    return X_train, y_train, X_test, y_test


def accuracy(model, X, y, batch_size=1000):
    correct = 0
    for i in range(0, len(X), batch_size):
        xb = Tensor(X[i : i + batch_size])
        logits = model(xb)
        preds = logits.data.argmax(axis=1)
        correct += (preds == y[i : i + batch_size]).sum()
    return correct / len(X)


def main():
    np.random.seed(SEED)

    X_train, y_train, X_test, y_test = load_mnist()
    print(f"train: {X_train.shape}, test: {X_test.shape}")

    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10),
    )
    optimizer = SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    n = len(X_train)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        perm = np.random.permutation(n)
        X_shuf, y_shuf = X_train[perm], y_train[perm]

        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, BATCH_SIZE):
            xb = Tensor(X_shuf[i : i + BATCH_SIZE])
            yb = y_shuf[i : i + BATCH_SIZE]

            logits = model(xb)
            loss = cross_entropy_loss(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.data)
            n_batches += 1

        train_acc = accuracy(model, X_train, y_train)
        test_acc = accuracy(model, X_test, y_test)
        history["train_loss"].append(epoch_loss / n_batches)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(
            f"epoch {epoch:2d}/{EPOCHS}  loss={epoch_loss / n_batches:.4f}  "
            f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
            f"({time.time() - t0:.1f}s)"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["train_loss"])
    ax1.set_title("Training loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")

    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["test_acc"], label="test")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("results/training_curves.png", dpi=150)
    print("Saved results/training_curves.png")
    print(f"Final test accuracy: {history['test_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()
