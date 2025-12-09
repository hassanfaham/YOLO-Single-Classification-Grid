from tkinter import filedialog
from logger_config import get_logger
logger = get_logger()

class ModelManager:
    def __init__(self, initial_model_path=None):
        self.models_list = []
        self.active_model = None 

        if initial_model_path:
            try:
                model_name = initial_model_path.split("/")[-1]
                self.models_list.append({"name": model_name, "path": initial_model_path})
                self.active_model = initial_model_path
            except Exception as e:
                logger.exception("Error initializing model manager with initial model path.")

    def upload_model(self):
        try:
            model_path = filedialog.askopenfilename(title="Select Model", filetypes=[("Model Files", "*.h5 *.pt *.pkl *.onnx")])
            if model_path:
                model_name = model_path.split("/")[-1]
                self.models_list.append({"name": model_name, "path": model_path})
                self.set_active_model(model_name)
                logger.success(f"Uploaded and selected model: {model_name}")
                return model_name
            else:
                logger.info("No model selected for upload.")
                return None
        except Exception as e:
            logger.exception("Error during model upload.")
            return None

    def set_active_model(self, model_name):
        try:
            model_entry = next((model for model in self.models_list if model["name"] == model_name), None)
            if model_entry:
                self.active_model = model_entry["path"]
            else:
                logger.warning(f"Model '{model_name}' not found in the uploaded models list.")
        except Exception as e:
            logger.exception("Error setting active model.")
