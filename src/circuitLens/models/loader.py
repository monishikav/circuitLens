from transformer_lens import HookedTransformer
class ModelLoader:
    def __init__(self,model_name="gpt2-small"):
        self.model_name = model_name
        self.model = None
    def load_model(self):
        print(f"Loading model:{self.model_name}")
        self.model = HookedTransformer.from_pretrained(
            self.model_name
        ) 
        print("Model loaded successfully")
        return self.model
    def get_model(self):
        if self.model is None:
            self.load_model()
        return self.model       