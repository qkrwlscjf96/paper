# 모델 학습

import json

import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def _build_model():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )


def model_training(df, check_cols, target_col):
    # Feature / Target 분리
    X = df[check_cols]
    y = df[target_col[0]]

    # 1차 분리: train:test = 7:3
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        shuffle=True,
        random_state=42,
        stratify=y,
    )

    # 2차 분리: train 내부를 6:1로 train/validation 분리
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=1 / 7,
        shuffle=True,
        random_state=42,
        stratify=y_train_full,
    )

    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

    # Train 데이터에 대해서만 Stratified K-Fold 적용
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(X_train, y_train), start=1):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        scaler = StandardScaler()
        X_fold_train_scaled = scaler.fit_transform(X_fold_train)
        X_fold_valid_scaled = scaler.transform(X_fold_valid)

        model = clone(_build_model())
        model.fit(X_fold_train_scaled, y_fold_train)
        y_fold_pred = model.predict(X_fold_valid_scaled)
        fold_accuracy = accuracy_score(y_fold_valid, y_fold_pred)
        cv_accuracies.append(float(fold_accuracy))
        print(f"Fold {fold_idx} Accuracy: {fold_accuracy}")

    print(f"K-Fold Accuracy Mean: {np.mean(cv_accuracies):.4f}")
    print(f"K-Fold Accuracy Std: {np.std(cv_accuracies):.4f}")

    # 최종 모델 학습
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_val_scaled = final_scaler.transform(X_val)
    X_test_scaled = final_scaler.transform(X_test)

    final_model = _build_model()
    final_model.fit(X_train_scaled, y_train)

    # Validation / Test 예측
    y_val_pred = final_model.predict(X_val_scaled)
    y_test_pred = final_model.predict(X_test_scaled)

    validation_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    confusion = confusion_matrix(y_test, y_test_pred)
    report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    report_text = classification_report(y_test, y_test_pred)

    print("Validation Accuracy:", validation_accuracy)
    print("Test Accuracy:", test_accuracy)
    print(confusion)
    print(report_text)

    return {
        "accuracy": float(test_accuracy),
        "validation_accuracy": float(validation_accuracy),
        "cv_accuracy_mean": float(np.mean(cv_accuracies)),
        "cv_accuracy_std": float(np.std(cv_accuracies)),
        "cv_fold_accuracies": cv_accuracies,
        "confusion_matrix": confusion.tolist(),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "model_params": json.loads(json.dumps(final_model.get_params(), default=str)),
    }
