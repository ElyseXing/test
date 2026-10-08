"""Classical baselines trained on the same EEG feature vectors."""

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_baselines(random_state: int = 42) -> dict[str, object]:
    return {
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,),
                max_iter=300,
                early_stopping=True,
                random_state=random_state,
            ),
        ),
    }
