from .kronos import KronosTokenizer, Kronos, KronosPredictor
from .kronos_classifier import KronosClassifier, KronosProbeClassifier

model_dict = {
    'kronos_tokenizer': KronosTokenizer,
    'kronos': Kronos,
    'kronos_predictor': KronosPredictor,
    'kronos_classifier': KronosClassifier,
    'kronos_probe_classifier': KronosProbeClassifier,
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        print(f"Model {model_name} not found in model_dict")
        raise NotImplementedError


