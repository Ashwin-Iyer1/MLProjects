"""A transparent, leakage-safe baseline using only the repository's labeled training CSV."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ['Age', 'Fare', 'FamilySize']
CATEGORICAL = ['Pclass', 'Sex', 'Embarked']


def prepare_features(frame):
    required = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    if not set(required).issubset(frame.columns):
        raise ValueError('Missing Titanic feature columns')
    if frame['PassengerId'].isna().any() or frame['PassengerId'].duplicated().any():
        raise ValueError('Passenger IDs must be present and unique')
    result = frame.copy()
    result['FamilySize'] = result['SibSp'] + result['Parch'] + 1
    return result[NUMERIC + CATEGORICAL]


def make_pipeline(seed=42):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)), ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return Pipeline([('prepare', ColumnTransformer([('numeric', numeric, NUMERIC), ('category', categorical, CATEGORICAL)])),
                     ('model', LogisticRegression(C=1.0, max_iter=2000, random_state=seed))])


def build_report(source=None, seed=42):
    source = Path(source) if source else ROOT / 'TitanicPred' / 'titanic' / 'train.csv'
    frame = pd.read_csv(source)
    if 'Survived' not in frame or set(frame['Survived'].unique()) != {0, 1}:
        raise ValueError('A labeled dataset with both binary classes is required')
    features, labels = prepare_features(frame), frame['Survived']
    train, holdout = train_test_split(np.arange(len(frame)), test_size=.2, stratify=labels, random_state=seed)
    model = make_pipeline(seed)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv = cross_val_score(model, features.iloc[train], labels.iloc[train], cv=folds, scoring='accuracy', n_jobs=1)
    model.fit(features.iloc[train], labels.iloc[train])
    actual = labels.iloc[holdout].to_numpy()
    probability = model.predict_proba(features.iloc[holdout])[:, 1]
    predicted = (probability >= .5).astype(int)
    baseline = DummyClassifier(strategy='most_frequent').fit(features.iloc[train], labels.iloc[train])
    coefficients = model.named_steps['model'].coef_[0]
    feature_names = model.named_steps['prepare'].get_feature_names_out()
    rows = []
    for index, value in zip(holdout, probability):
        row = frame.iloc[index]
        rows.append({'id': int(row.PassengerId), 'actual': int(row.Survived), 'probability': float(value),
                     'sex': row.Sex, 'class': int(row.Pclass), 'age': None if pd.isna(row.Age) else float(row.Age)})
    # Canonical LF content hash is reproducible on Windows and Linux checkouts.
    digest = hashlib.sha256(source.read_text(encoding='utf-8').replace('\r\n', '\n').encode('utf-8')).hexdigest()
    return {'schemaVersion': 1, 'experiment': 'Titanic logistic baseline', 'seed': seed,
            'source': {'path': 'TitanicPred/titanic/train.csv', 'sha256LF': digest, 'rows': len(frame),
                       'note': 'Existing labeled Kaggle Titanic training data. answer.csv and competition test.csv are not evaluated.'},
            'split': {'method': 'Stratified 80/20 holdout; preprocessing fitted on training only', 'trainIds': frame.iloc[train].PassengerId.astype(int).tolist(),
                      'holdoutIds': frame.iloc[holdout].PassengerId.astype(int).tolist()},
            'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'scikitLearn': sklearn.__version__},
            'model': {'type': 'L2 logistic regression', 'C': 1.0, 'maxIterations': 2000, 'iterations': int(model.named_steps['model'].n_iter_[0]),
                      'numeric': NUMERIC, 'categorical': CATEGORICAL,
                      'trainingMedians': dict(zip(NUMERIC, model.named_steps['prepare'].named_transformers_['numeric'].named_steps['impute'].statistics_.tolist())),
                      'coefficients': [{'feature': name, 'coefficient': float(value)} for name, value in zip(feature_names, coefficients)]},
            'metrics': {'threshold': .5, 'accuracy': float(accuracy_score(actual, predicted)), 'balancedAccuracy': float(balanced_accuracy_score(actual, predicted)),
                        'rocAuc': float(roc_auc_score(actual, probability)), 'brier': float(brier_score_loss(actual, probability)),
                        'logLoss': float(log_loss(actual, probability)), 'confusion': confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
                        'majorityBaselineAccuracy': float(baseline.score(features.iloc[holdout], labels.iloc[holdout]))},
            'trainingCV': {'folds': 5, 'accuracy': cv.tolist(), 'mean': float(cv.mean()), 'standardDeviation': float(cv.std(ddof=1))},
            'predictions': sorted(rows, key=lambda row: row['id'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'lab' / 'report.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    report = build_report(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'holdoutRows': len(report['predictions']), 'accuracy': report['metrics']['accuracy'], 'rocAuc': report['metrics']['rocAuc'], 'trainingCV': report['trainingCV']['mean'], 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
