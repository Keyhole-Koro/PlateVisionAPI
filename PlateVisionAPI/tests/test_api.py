import asyncio
import httpx
import os
import time

API_URL = "http://127.0.0.1:8000/process_image/"
TEST_IMAGE_PATH = "tests/test_image.jpg"  # Replace with the path to your test image

async def test_process_image():
    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"Test image not found at {TEST_IMAGE_PATH}")

    with open(TEST_IMAGE_PATH, "rb") as image_file:
        image_data = image_file.read()

    retries = 5
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient() as client:
                start_time = time.time()
                # Add the `measure=true` parameter to the request
                response = await client.post(
                    API_URL,
                    files={"file": ("test_image.jpg", image_data, "image/jpeg")},
                    params={"measure": "true", "return_image_annotation": "false"}  # Add query parameters here
                )
                elapsed_time = time.time() - start_time
                # Save the response image if it
            assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
            if "image" in response.headers.get("content-type", ""):
                with open("response_image.jpg", "wb") as output_file:
                    output_file.write(response.content)
                print("Response image saved as 'response_image.jpg'")
            response_data = response.json()
            assert "result" in response_data, "Response does not contain 'result'"
            print(f"Test passed. Response: {response_data}, Time taken: {elapsed_time:.2f} seconds")
            return
        except httpx.ConnectError:
            print(f"Connection attempt {attempt + 1} failed. Retrying...")
            await asyncio.sleep(2)  # Wait before retrying

    raise RuntimeError("Failed to connect to the API after multiple attempts.")

if __name__ == "__main__":
    asyncio.run(test_process_image())