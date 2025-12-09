import sys, traceback, os

LOG_FILE = "fatal_error_log.txt"

def write_fatal_error(e_type, e_value, e_traceback):
    """
    Writes any unhandled exception to fatal_error_log.txt.
    Safe for threads, Tkinter, and app-wide exceptions.
    """
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n===============================\n")
            f.write("UNHANDLED FATAL ERROR\n")
            f.write("===============================\n")
            traceback.print_exception(e_type, e_value, e_traceback, file=f)
            f.write("\n")
    except Exception:
        pass

sys.excepthook = write_fatal_error

def install_tkinter_error_hook():
    """
    Patch Tkinter so UI callback errors are also logged.
    """
    try:
        import tkinter
    except ImportError:
        return

    def tk_callback_error(type_, value_, traceback_):
        write_fatal_error(type_, value_, traceback_)

    tkinter.Tk.report_callback_exception = tk_callback_error
