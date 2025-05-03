import os
import psutil
import time
import asyncio
from functools import wraps

def monitor_memory():
    """ Monitor the memory usage in MB. """
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB
    return memory_usage

def measure_time_and_memory(func=None, *, enabled=True):
    """Decorator to measure execution time and peak memory usage, controlled by a flag."""
    if func is None:
        return lambda f: measure_time_and_memory(f, enabled=enabled)

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        if not enabled:
            return await func(*args, **kwargs)

        start_time = time.time()
        max_memory = monitor_memory()  # Initialize max memory

        async def monitor_peak_memory():
            nonlocal max_memory
            while True:
                current_memory = monitor_memory()
                max_memory = max(max_memory, current_memory)
                await asyncio.sleep(0.1)  # Check memory every 100ms

        # Start monitoring memory in the background
        memory_task = asyncio.create_task(monitor_peak_memory())

        # Run the actual function
        result = await func(*args, **kwargs)

        # Stop memory monitoring
        memory_task.cancel()
        try:
            await memory_task
        except asyncio.CancelledError:
            pass

        total_time = time.time() - start_time
        print(f"{func.__name__} Execution Time: {total_time:.4f} seconds")
        print(f"{func.__name__} Peak Memory Usage: {max_memory:.2f} MB")
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        if not enabled:
            return func(*args, **kwargs)

        start_time = time.time()
        max_memory = monitor_memory()  # Initialize max memory

        def monitor_peak_memory():
            nonlocal max_memory
            while True:
                current_memory = monitor_memory()
                max_memory = max(max_memory, current_memory)
                time.sleep(0.1)  # Check memory every 100ms

        # Start monitoring memory in the background
        import threading
        memory_thread = threading.Thread(target=monitor_peak_memory, daemon=True)
        memory_thread.start()

        # Run the actual function
        result = func(*args, **kwargs)

        # Stop memory monitoring
        memory_thread.join(timeout=0)

        total_time = time.time() - start_time
        print(f"{func.__name__} Execution Time: {total_time:.4f} seconds")
        print(f"{func.__name__} Peak Memory Usage: {max_memory:.2f} MB")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
