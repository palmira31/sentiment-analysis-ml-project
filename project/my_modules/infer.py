import os

import hydra
import joblib
import onnxruntime as ort
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    data_dir = to_absolute_path(cfg.inference.data_dir)
    model_dir = to_absolute_path(cfg.inference.model_dir)
    output_dir = to_absolute_path(cfg.inference.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    vectorizer_path = os.path.join(model_dir, cfg.inference.vectorizer_file)
    label_encoder_path = os.path.join(model_dir, cfg.inference.label_encoder_file)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)

    test_path = os.path.join(data_dir, cfg.inference.test_file)
    df_test = pd.read_csv(test_path)

    X_test = vectorizer.transform(df_test["sentence"]).toarray()

    onnx_model_path = os.path.join(model_dir, cfg.inference.onnx_model_file)
    ort_session = ort.InferenceSession(onnx_model_path)

    inputs = {ort_session.get_inputs()[0].name: X_test.astype("float32")}

    outputs = ort_session.run(None, inputs)
    preds = outputs[0]

    pred_labels = label_encoder.inverse_transform(preds.argmax(axis=1))

    df_out = df_test.copy()
    df_out["predicted_label"] = pred_labels
    output_path = os.path.join(output_dir, "test_predictions.csv")
    df_out.to_csv(output_path, index=False)

    print(f"Inference done. Results saved to {output_path}")


if __name__ == "__main__":
    main()
