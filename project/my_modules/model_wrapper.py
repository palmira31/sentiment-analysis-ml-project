import mlflow.pyfunc
import torch


class SentimentClassifierWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, vectorizer, label_encoder, model=None):
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.model = model

    def load_context(self, context):
        """Load the model and artifacts when serving"""
        if self.model is None:
            self.model = torch.load(
                context.artifacts["model_path"],
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )
        self.model.eval()

    def predict(self, context, model_input):
        """
        Args:
            context: MLflow context
            model_input: pandas DataFrame with 'text' column
        Returns:
            numpy array with predicted labels
        """
        features = self.vectorizer.transform(model_input["text"].values)
        features_tensor = torch.FloatTensor(features.toarray())

        with torch.no_grad():
            if torch.cuda.is_available():
                features_tensor = features_tensor.cuda()
                self.model = self.model.cuda()

            predictions = self.model(features_tensor)

        predicted_classes = predictions.argmax(dim=1).cpu().numpy()
        labels = self.label_encoder.inverse_transform(predicted_classes)

        return labels
