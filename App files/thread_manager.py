import threading

class ThreadManager:
    def __init__(self):
        self.threads = []

    def run_in_thread(self, task, callback=None):
        def thread_function():
            task()
            if callback:
                callback()

        thread = threading.Thread(target=thread_function)
        thread.start()
        self.threads.append(thread)

    def join_all_threads(self):
        for thread in self.threads:
            thread.join()