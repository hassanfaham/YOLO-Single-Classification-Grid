from fatal_error_handler import install_tkinter_error_hook
install_tkinter_error_hook()


import customtkinter as ctk
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import json
from threading import Event
from datetime import datetime
import traceback
from logger_config import get_logger
logger = get_logger()

from thread_manager import ThreadManager
from model_manager import ModelManager
from processing_manager import ProcessingManager


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit")

logo_ico = "resources/images/logo.ico"
logo_light = "resources/images/FastDoc-Light-Mode.png"

text_color_sucesso = '#800080'
button_color_sucesso = '#800080'
button_hover_color_sucesso = '#4B0082'
color_light_grey_button = '#b0a5d8'
dark_grey_button = '#4c4084'
light_blue_color = "#9370DB"
dark_blue_color = "#4B0082"

class TestApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        try:
            with open("config.json", "r") as config_file:
                config = json.load(config_file)
            logger.success("Configuration loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load configuration.")
            raise
        
        self.watch_single_folder_path = config["watch_single_folder_path"]  
        initial_model_path = config.get("initial_model_path")
        self.grid_config = config["grid_config"]
        self.rows = self.grid_config["rows"]
        self.columns = self.grid_config["columns"]
        self.total_pieces = self.grid_config["total_pieces"]
        self.enable_grid = config["enable_grid"]


        self.model_manager = ModelManager(initial_model_path) 
        self.thread_manager = ThreadManager()

        self.image_doc = None
        self.shutdown_event = Event()


        self.total_components = 0
        self.ok_components = 0
        self.nok_components = 0

        self.total_components_label = None
        self.ok_components_label = None
        self.nok_components_label = None

        self.grid_data = {}

        self.title("Inspection App")
        self.iconbitmap(logo_ico)
        self.geometry(self.center_window(1400, 900))
        self.attributes("-topmost", True)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("./purple.json")

        self.bind("<Escape>", lambda e: self.exit_app())
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.sidebar_VISIBLE = False

        self.create_widgets()
        self.state("zoomed")
        


        if initial_model_path:
            self.update_model_dropdown()
            self.start_monitoring_and_processing()
        else:
                self.create_monitoring_buttons()

    def exit_app(self):
        logger.info("Exiting application and stopping all threads...")
        self.shutdown_event.set()

        try:
            self.processing_manager.stop_monitoring()
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")


        try:
            self.thread_manager.join_all_threads()
        except Exception:
            logger.warning("Some threads could not be joined.")

        self.destroy()
        os._exit(0)


    def center_window(self, width, height):
        screen_width, screen_height = self.winfo_screenwidth(), self.winfo_screenheight()
        x_coordinate = (screen_width // 2) - (width // 2)
        y_coordinate = (screen_height // 2) - (height // 2)
        return f"{width}x{height}+{x_coordinate}+{y_coordinate}"

    def create_widgets(self):
        self.sidebar_esq = self.create_sidebar()
        self.tabview1 = self.create_tabs()

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.btn_detection()

    def create_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=100, corner_radius=8)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=10, pady=10, rowspan=2)

        self.detection_button = ctk.CTkButton(sidebar, text="Image Detection", command=self.btn_detection,
                                              fg_color=light_blue_color, hover_color=dark_blue_color, width=80)
        self.detection_button.pack(pady=20, padx=10)
        
        ctk.CTkOptionMenu(sidebar, values=["Dark", "Light", "System"], command=self.change_appearance_mode).pack(side="bottom", padx=10, pady=(0, 10))
        ctk.CTkLabel(sidebar, text="Choose the theme:").pack(side="bottom",padx=10, pady=(5, 0))

        return sidebar

    def create_tabs(self):
        return self.create_tab_view_1()

    def create_tab_view_1(self):
        main_frame1 = ctk.CTkFrame(self)
        main_frame1.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)

        main_frame1.grid_rowconfigure(0, weight=0)
        main_frame1.grid_rowconfigure(1, weight=1)
        main_frame1.grid_columnconfigure(0, weight=3) 
        main_frame1.grid_columnconfigure(1, weight=1) 


        self.create_buttons_section(main_frame1)
        self.create_image_section(main_frame1)
        self.create_squares_section(main_frame1)

        return main_frame1

    def create_buttons_section(self, frame):

        self.buttons_frame = ctk.CTkFrame(frame)
        self.buttons_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 5))

        self.buttons_frame.grid_columnconfigure(0, weight=1)
        self.buttons_frame.grid_columnconfigure(1, weight=1)
        self.buttons_frame.grid_columnconfigure(2, weight=2)
        self.buttons_frame.grid_columnconfigure(3, weight=1)
        self.buttons_frame.grid_columnconfigure(4, weight=1)

        self.buttons_frame.grid_rowconfigure(0, weight=1)
        self.buttons_frame.grid_rowconfigure(1, weight=1)

        self.browse_button = ctk.CTkButton(
            self.buttons_frame, text="Upload Model", command=self.upload_model,
            width=100, fg_color=button_color_sucesso, hover_color=button_hover_color_sucesso
        )
        self.browse_button.grid(row=0, column=2, padx=10, pady=5, sticky="n")

        self.model_var = ctk.StringVar(value="No model selected")
        self.model_dropdown = ctk.CTkOptionMenu(
            self.buttons_frame, variable=self.model_var, values=["No models yet"],
            width=150, command=self.select_model
        )
        self.model_dropdown.grid(row=1, column=2, padx=10, pady=5, sticky="n")

        if self.enable_grid:
            self.result_button = ctk.CTkButton(
                self.buttons_frame,
                text="Show Result (Processing...)",
                command=self.show_result_popup,
                width=100,
                fg_color="red",
                hover_color="#FF6347"
            )
            self.result_button.grid(row=0, column=3, rowspan=2, padx=10, pady=5, sticky="e")


    def create_image_section(self, frame):
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=5)
        frame.grid_columnconfigure(0, weight=6)
        frame.grid_columnconfigure(1, weight=4)

        self.image_frame = ctk.CTkFrame(frame)
        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(10, 40))
            
        self.canvas = ctk.CTkCanvas(self.image_frame, width=800, height=600, bg="#252627")
        self.canvas.pack(expand=True, fill="both")
        
    def create_squares_section(self, frame):
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        squares_frame = ctk.CTkFrame(frame)
        squares_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 5), pady=(30, 40))
        squares_frame.grid_columnconfigure(0, weight=1)

        for r in range(8):
            squares_frame.grid_rowconfigure(r, weight=1)

        items = [
            ("Total Components", "lightblue"),
            ("Ok", "lightgreen"),
            ("Nok", "lightcoral")
        ]

        row_index = 1
        for label, color in items:
            self.create_square(squares_frame, label, color, row_index)
            row_index += 2

        self.reset_button = ctk.CTkButton(
            squares_frame, text="Reset Count",
            command=self.reset_counters,
            width=120, fg_color="red", hover_color="#FF6347"
        )
        self.reset_button.grid(row=0, column=0, pady=(40, 20), sticky="n")


    def create_square(self, frame, label_text, color, row):
        ctk.CTkLabel(frame, text=label_text).grid(row=row, column=0, padx=10, pady=(10, 5))

        square = ctk.CTkFrame(frame, fg_color=color)
        square.grid(row=row + 1, column=0, padx=10, pady=(0, 40), sticky="nsew")
        square.grid_propagate(False)
        square.configure(width=140, height=120)

        count_label = ctk.CTkLabel(square, text="0000000",
                                anchor="center", text_color="black",
                                font=("Arial", 24))
        count_label.pack(pady=(10, 0))

        percent_label = ctk.CTkLabel(square, text="0%",
                                    anchor="center", text_color="black",
                                    font=("Arial", 16))
        percent_label.pack(pady=(0, 10))

        if label_text == "Total Components":
            self.total_components_label = count_label
            self.total_percent_label = percent_label
        elif label_text == "Ok":
            self.ok_components_label = count_label
            self.ok_percent_label = percent_label
        elif label_text == "Nok":
            self.nok_components_label = count_label
            self.nok_percent_label = percent_label


    def btn_detection(self):
        self.tabview1.tkraise()
        self.update_button_colors(self.detection_button)

    def update_button_colors(self, active_button):
        buttons = [self.detection_button]
        for button in buttons:
            button.configure(fg_color=light_blue_color if button == active_button else color_light_grey_button)

    def change_appearance_mode(self, mode):
        ctk.set_appearance_mode(mode.lower())

    def upload_model(self):
        self.thread_manager.run_in_thread(self.upload_model_task, self.update_model_dropdown)

    def upload_model_task(self):
        model_name = self.model_manager.upload_model()
        if model_name:
            self.model_manager.set_active_model(model_name)

    def update_model_dropdown(self):
        model_names = [model["name"] for model in self.model_manager.models_list]
        self.model_dropdown.configure(values=model_names)
        self.model_var.set(model_names[-1])

    def select_model(self, selected_model):
        logger.info(f"Selected model: {selected_model}")
        self.model_manager.set_active_model(selected_model)

    def start_monitoring_and_processing(self):
        active_model_path = self.model_manager.active_model


        self.processing_manager = ProcessingManager(
                watch_single_folder_path=self.watch_single_folder_path,
                model_path=active_model_path,
                update_callback=self.update_component_counters,
                shutdown_event=self.shutdown_event,
                update_grid_callback=self.update_grid_data,
 
            )
        self.thread_manager.run_in_thread(self.processing_manager.start_monitoring)
        self.thread_manager.run_in_thread(lambda: self.processing_manager.process_single_image(self.display_image_callback))



    def start_monitoring_button_clicked(self):
        if not self.model_manager.active_model:
            logger.error("No model loaded. Please upload a model first.")
            return

        self.start_monitoring_and_processing()
        self.start_monitoring_button.configure(fg_color="#4CAF50", hover_color="#81C784")
        self.stop_monitoring_button.configure(fg_color=button_color_sucesso, hover_color=button_hover_color_sucesso)

    def stop_monitoring_button_clicked(self):
        self.processing_manager.stop_monitoring()
        logger.info("Stopped monitoring.")

        self.start_monitoring_button.configure(fg_color=button_color_sucesso, hover_color=button_hover_color_sucesso)
        self.stop_monitoring_button.configure(fg_color="#FF6666", hover_color="#FF9999")


    def create_monitoring_buttons(self):
        self.start_monitoring_button = ctk.CTkButton(
            self.buttons_frame, text="Start Monitoring", command=self.start_monitoring_button_clicked,
            width=120, fg_color=button_color_sucesso, hover_color=button_hover_color_sucesso
        )
        self.start_monitoring_button.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        self.stop_monitoring_button = ctk.CTkButton(
            self.buttons_frame, text="Stop Monitoring", command=self.stop_monitoring_button_clicked,
            width=120, fg_color=button_color_sucesso, hover_color=button_hover_color_sucesso
        )
        self.stop_monitoring_button.grid(row=1, column=1, padx=10, pady=5, sticky="w")

    def display_image_on_canvas(self, image, canvas):
        try:
            logger.info("Displaying image on canvas...")
            self.original_image = image.copy()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            image_width, image_height = image.size
            image_aspect_ratio = image_width / image_height
            canvas_aspect_ratio = canvas_width / canvas_height

            if canvas_aspect_ratio > image_aspect_ratio:
                new_height = canvas_height
                new_width = int(image_aspect_ratio * new_height)
            else:
                new_width = canvas_width
                new_height = int(new_width / image_aspect_ratio)

            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            tk_image = ImageTk.PhotoImage(resized_image)
            canvas.image = tk_image
            canvas.delete("all")
            canvas.create_image(canvas_width // 2, canvas_height // 2, image=tk_image, anchor="center")
            logger.info("Displayed processed image.")
        except Exception as e:
            logger.exception("Error displaying image")


    def display_image_callback(self, image):
        self.display_image_on_canvas(image, self.canvas)

    def update_component_counters(self, piece_status):
        try:
            self.total_components += 1
            total = self.total_components

            self.total_components_label.configure(text=str(total).zfill(7))

            if piece_status == "nok":
                self.nok_components += 1
                self.nok_components_label.configure(text=str(self.nok_components).zfill(7))
            elif piece_status == "ok":
                self.ok_components += 1
                self.ok_components_label.configure(text=str(self.ok_components).zfill(7))
            else:
                self.nok_components += 1
                self.nok_components_label.configure(text=str(self.nok_components).zfill(7))

            self.update_percentages()

        except Exception:
            logger.exception("Error updating component counters")


    def update_percentages(self):
        total = max(self.total_components, 1)

        ok_pct = (self.ok_components / total) * 100
        nok_pct = (self.nok_components / total) * 100

        self.ok_percent_label.configure(text=f"{ok_pct:.1f}%")
        self.nok_percent_label.configure(text=f"{nok_pct:.1f}%")

        self.total_percent_label.configure(text="100%")




    def reset_counters(self):
        self.total_components = 0
        self.ok_components = 0
        self.nok_components = 0

        self.total_components_label.configure(text="0000000")
        self.ok_components_label.configure(text="0000000")
        self.nok_components_label.configure(text="0000000")

        self.total_percent_label.configure(text="0%")
        self.ok_percent_label.configure(text="0%")
        self.nok_percent_label.configure(text="0%")

        logger.info("All counters and percentages have been reset to zero.")



    def show_result_popup(self):
        if not self.enable_grid:
            logger.warning("Grid display is disabled in configuration.")
            return

        if not hasattr(self, "result_popup") or not self.result_popup.winfo_exists():
            self.result_popup = ctk.CTkToplevel(self)
            self.result_popup.title("Inspection Results")
            self.result_popup.geometry("800x600")
            self.result_popup.attributes("-topmost", True)

            self.result_popup.lift()
            self.result_popup.focus_force()
            self.result_popup.grab_set()

            self.grid_frame = ctk.CTkFrame(self.result_popup)
            self.grid_frame.pack(expand=True, fill="both", padx=10, pady=10)

            self.grid_widget_map = {}

            for r in range(self.rows):
                self.grid_frame.grid_rowconfigure(r, weight=1)
            for c in range(self.columns):
                self.grid_frame.grid_columnconfigure(c, weight=1)

            for row in range(self.rows):
                for col in range(self.columns):
                    lbl = ctk.CTkLabel(
                        self.grid_frame,
                        text="undetected",
                        fg_color="gray",
                        font=("Arial", 14, "bold"),
                        anchor="center"
                    )
                    lbl.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
                    self.grid_widget_map[(row, col)] = lbl

            def on_close():
                logger.info("Closing result popup (grid stays alive).")
                self.result_popup.destroy()

            self.result_popup.protocol("WM_DELETE_WINDOW", on_close)

        self.refresh_result_popup()





    def update_grid_data(self, data):
        if not self.enable_grid:
            logger.warning("Grid logic is disabled in configuration.")
            return

        if isinstance(data, dict) and "status" not in data:
            logger.info("Received legacy grid data format, treating as complete palette.")
            self.grid_data = data
            if hasattr(self, "result_button"):
                self.result_button.configure(
                    fg_color="green",
                    hover_color="#00A651",
                    text="Show Result (Complete)"
                )
            return

        status = data.get("status")
        logger.info(f"Received grid update message: {status}")

        if status == "start_new_palette":
            self.grid_data = {}
            if hasattr(self, "result_button"):
                self.result_button.configure(
                    fg_color="red",
                    hover_color="#FF6347",
                    text="Show Result (Processing...)"
                )
            return

        if status == "update_cell":
            position = tuple(data["position"])
            piece_status = data["piece_status"]
            self.grid_data[position] = piece_status

            if hasattr(self, "result_popup") and self.result_popup.winfo_exists():
                self.refresh_result_popup()
            return

        if status == "palette_complete":
            self.grid_data = data.get("grid", {})
            if hasattr(self, "result_button"):
                self.result_button.configure(
                    fg_color="green",
                    hover_color="#00A651",
                    text="Show Result (Complete)"
                )

            if hasattr(self, "result_popup") and self.result_popup.winfo_exists():
                self.refresh_result_popup()
            return



    def refresh_result_popup(self):
        if not hasattr(self, "result_popup") or not self.result_popup.winfo_exists():
            logger.warning("Result popup does not exist.")
            return
        if not hasattr(self, "grid_widget_map"):
            logger.warning("Grid widget map not found.")
            return

        logger.info("Refreshing result popup grid...")
        status_colors = {
            "ok": "#4CAF50",
            "nok": "#FF6666",
            "undetected": "gray",
        }

        for row in range(self.rows):
            for col in range(self.columns):
                status = self.grid_data.get((row, col), "undetected")
                widget = self.grid_widget_map[(row, col)]

                try:
                    widget.configure(
                        text=f"{status}\n({row},{col})",
                        fg_color=status_colors.get(status, "gray")
                    )
                except Exception:
                    logger.exception("Error updating result popup grid.")




def start_application():
    try:
        app = TestApp()
        app.mainloop()
    except KeyboardInterrupt:
        try:
            app.exit_app()
        except:
            os._exit(0)
    except Exception as e:
        with open("error_log.txt", "a") as log_file:
            log_file.write("An unexpected error occurred:\n")
            log_file.write(traceback.format_exc())
            log_file.write("\n" + "=" * 50 + "\n")
        logger.critical("An unexpected error occurred. Check 'error_log.txt' for details.")
        os._exit(1)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support() 
    start_application()





