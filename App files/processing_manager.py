
from fatal_error_handler import install_tkinter_error_hook
install_tkinter_error_hook()


from logger_config import get_logger
logger = get_logger()


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from queue import Queue
import threading
import time
import os
from PIL import ImageTk, Image, ImageFile
import PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import json
import os
import numpy as np
import time
from collections import OrderedDict
from threading import Lock





class ImageHandler(FileSystemEventHandler):
    def __init__(self, image_queue, debounce_seconds=1.0, max_cache_size=100):
        self.image_queue = image_queue
        self.debounce_seconds = debounce_seconds
        self.max_cache_size = max_cache_size
        self.recent_events = OrderedDict()
        self.lock = Lock()

    def handle_event(self, event):
        path = event.src_path
        now = time.time()

        if event.is_directory or not path.lower().endswith(('.jpg', '.png', '.jpeg')):
            return

        with self.lock:
            self._cleanup_old_entries(now)
            last_time = self.recent_events.get(path)
            if last_time and now - last_time < self.debounce_seconds:
                return 

            self.recent_events[path] = now
            if len(self.recent_events) > self.max_cache_size:
                self.recent_events.popitem(last=False)

        logger.info(f"Image event detected: {path}")
        for attempt in range(5):
            if self.is_image_fully_written(path):
                try:
                    self.image_queue.put(path, block=False)
                    logger.info(f"Image queued for processing: {path}")
                except Exception as e:
                    logger.warning(f"Failed to enqueue image: {e}")
                break
            else:
                logger.warning(f"[{attempt+1}/5] File not ready yet: {path}")
                time.sleep(0.3)
        else:
            logger.error(f"Image was never ready after 5 attempts: {path}")

    def on_created(self, event):
        self.handle_event(event)

    def on_modified(self, event):
        self.handle_event(event)

    def is_image_fully_written(self, path, wait_time=1.0, interval=0.2):
        end_time = time.time() + wait_time
        last_size = -1

        while time.time() < end_time:
            try:
                current_size = os.path.getsize(path)
                if current_size == last_size:
                    with open(path, "rb") as f:
                        f.read()
                    return True
                last_size = current_size
            except OSError:
                pass
            time.sleep(interval)

        return False

    def _cleanup_old_entries(self, now):
        expired_keys = [k for k, t in self.recent_events.items() if now - t > self.debounce_seconds]
        for key in expired_keys:
            del self.recent_events[key]

       

class ProcessingManager:
    def __init__(self, 
                 watch_single_folder_path=None, 
                 model_path=None, 
                 update_callback=None, 
                 shutdown_event=None, 
                 update_grid_callback=None, 

            ):
        
        self.model_path = model_path
        self.observer = Observer()
        self.keep_processing = True
        self.model = None
        self.update_callback = update_callback
        self.update_batch_callback = update_grid_callback




        self.shutdown_event = shutdown_event
        self.processing_active = False 
        


        try:
            with open("config.json", "r") as config_file:
                config = json.load(config_file)
            self.prediction_parameters = config["prediction_parameters"]
            self.plotting_parameters = config["plotting_parameters"]
            self.status_logic = config["status_logic"]

            grid_config = config["grid_config"]
            self.rows = grid_config["rows"]
            self.columns = grid_config["columns"]
            self.total_pieces = grid_config["total_pieces"]

            self.enable_grid = config["enable_grid"]

            self.queue_size = config["queue_size"]

        except Exception as e:
            logger.exception("Failed to load configuration")
            raise

        
        self.watch_single_folder_path = watch_single_folder_path
        self.single_image_queue = Queue(maxsize=self.queue_size)




    def load_model(self):
        from ultralytics import YOLO

        try:
                logger.info(f"Initializing YOLO model using the path: {self.model_path}")
                self.model = YOLO(self.model_path)
                logger.info("YOLO model loaded successfully.")

        except Exception as e:
                logger.exception("Failed to load the models")
                raise


    def start_monitoring(self):

        event_handler = ImageHandler(self.single_image_queue)
        self.observer.schedule(event_handler, self.watch_single_folder_path, recursive=False)
        logger.info("Started monitoring the folder for new images...")
        threading.Thread(target=lambda:self.monitor_queue(self.watch_single_folder_path, self.single_image_queue), daemon=True).start()

        self.observer.start()

    def stop_monitoring(self):
        logger.info("Stopping observer...")

        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()

        self.keep_processing = False



    def process_image_core(self, image_path):

        self.processing_active = True
        logger.info(f"Processing image: {os.path.basename(image_path)}")
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}. Skipping processing.")
            return None, "nok"
        
        try:

            image = Image.open(image_path)

            results = self.model.predict(
                    source=image,
                    iou=self.prediction_parameters.get("iou", 0.5),
                    conf=self.prediction_parameters.get("conf", 0.6),
                    classes=self.prediction_parameters.get("classes"),
                    save=False
                )

            result = results[0]
            print(result)
            piece_status = None
            predicted_image = None
                
            if hasattr(result, "probs") and result.probs is not None:
                    probs = result.probs.data.cpu().numpy()
                    class_id = int(np.argmax(probs))
                    class_name = self.model.names[class_id]
                    confidence = probs[class_id]

                    for status, keywords in self.status_logic.items():
                        if any(keyword in class_name.lower() for keyword in keywords):
                            piece_status = status
                            break

                    image_np = np.array(image)
                    label_color = (0, 255, 0) if piece_status == "ok" else (255, 0, 0)
                    thickness = 40

                    h, w, _ = image_np.shape
                    cv2.rectangle(image_np, (0, 0), (w - 1, h - 1), label_color, thickness)

                    predicted_image = Image.fromarray(image_np)
            else:
                    logger.warning("Unknown YOLO model output type. Defaulting to NOK.")
                    predicted_image = image
                
            self.processing_active = False
            
            return predicted_image, piece_status
        
        except (OSError, IOError, PIL.UnidentifiedImageError) as e:
            logger.error(f"Error processing image {image_path}: {e}")
            os.remove(image_path)
            return None, "nok"

    def process_single_image(self, display_image_callback):
        """Process images in single mode."""
        try:
            if self.model is None and not self.shutdown_event.is_set():
                self.load_model()

            processed_results = {}
            cell_positions = self.generate_cell_positions()
            processed_count = 0

            while self.keep_processing and not self.shutdown_event.is_set():
                if not self.single_image_queue.empty():
                    try:
                        image_path = self.single_image_queue.get()

                        if not os.path.exists(image_path):
                            logger.error(f"Image not found: {image_path}")
                            continue
                        predicted_image, piece_status = self.process_image_core(image_path)

                        logger.info(f"Processed image status: {piece_status}")

                        if predicted_image:
                            display_image_callback(predicted_image)
                        else:
                            logger.warning("No processed image available to display.")

                        if predicted_image is not None and piece_status is not None:
                            display_image_callback(predicted_image)
                            self.update_callback(piece_status)
                        
                        if self.enable_grid:
                            if processed_count == 0:
                                processed_results.clear()
                                try:
                                    self.update_batch_callback({
                                        "status": "start_new_palette"
                                    })
                                except Exception as e:
                                    logger.exception("Error sending 'start_new_palette' grid update")

                            if processed_count < len(cell_positions):
                                position = cell_positions[processed_count]
                            else:
                                processed_count = 0
                                processed_results.clear()
                                try:
                                    self.update_batch_callback({
                                        "status": "start_new_palette"
                                    })
                                except Exception as e:
                                    logger.exception("Error sending 'start_new_palette' grid update after overflow")

                                position = cell_positions[processed_count]
                            processed_results[position] = piece_status

                            try:
                                self.update_batch_callback({
                                    "status": "update_cell",
                                    "position": position,
                                    "piece_status": piece_status,
                                    "grid": processed_results.copy(),
                                    "count": processed_count + 1
                                })
                            except Exception as e:
                                logger.exception("Error sending 'update_cell' grid update")

                            processed_count += 1

                            if processed_count == self.total_pieces:
                                try:
                                    self.update_batch_callback({
                                        "status": "palette_complete",
                                        "grid": processed_results.copy(),
                                        "count": processed_count
                                    })
                                except Exception as e:
                                    logger.exception("Error sending 'palette_complete' grid update")

                                processed_count = 0
                                processed_results.clear()




                    except FileNotFoundError as e:
                        logger.error(f"File not found: {e}")
                        continue
                    except Exception as e:
                        logger.exception(f"Unexpected error processing image: {e}")
                        continue

                time.sleep(0.5)
        except Exception as e:
            logger.exception("Unexpected error during image processing")
            
    def generate_cell_positions(self):
        positions = []
        for row in range(self.rows):
            if row % 2 == 0:
                positions.extend([(row, col) for col in range(self.columns)])
            else:
                positions.extend([(row, col) for col in reversed(range(self.columns))])
        return positions

    def monitor_queue(self, folder_to_watch, image_queue):
        inactivity_start = time.time()
        while self.keep_processing and not self.shutdown_event.is_set():
            if not any(filename.lower().endswith(('.jpg', '.png', '.jpeg')) for filename in os.listdir(folder_to_watch)):
                time.sleep(1)
                continue

