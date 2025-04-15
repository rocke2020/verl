import ray

if not ray.is_initialized():
    print("Ray is not initialized. Please start Ray first.")
else:
    print("Ray is initialized.")
    