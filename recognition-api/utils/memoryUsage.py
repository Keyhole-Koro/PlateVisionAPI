import os
import psutil
import time

def monitor_memory():
    """ Monitor the memory usage in MB. """
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB
    return memory_usage


def measure_time(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        total_time = time.time() - start_time
        print(f"{func.__name__} Execution Time: {total_time:.4f} seconds")
        return result
    return wrapper
