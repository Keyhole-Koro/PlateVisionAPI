import asyncio
import httpx
import os
import time
import base64

API_URL = "https://4yosom3xj2.execute-api.ap-northeast-3.amazonaws.com/stage/process_image"  # Use the correct address
API_KEY = "57b0992f"  # Replace with your actual API key
TEST_IMAGE_PATH = "tests/test_image.jpg"

async def test_process_image():
    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"Test image not found at {TEST_IMAGE_PATH}")

    with open(TEST_IMAGE_PATH, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode()

    json_payload = {
        "base64_data": {
            "isBase64Encoded": True,
            "body": image_data
        }
    }

    retries = 5
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1}: Sending request to {API_URL}")
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:  # Increased timeout to 120 seconds
                start_time = time.time()
                response = await client.post(
                    API_URL,
                    json=json_payload,
                    params={"measure": "true", "return_image_annotation": "false"},
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                elapsed_time = time.time() - start_time

                print(f"Response received in {elapsed_time:.2f} seconds")
                assert response.status_code == 200, f"Unexpected status code: {response.status_code}...{response}"
                print(f"Test passed. Response: {response.json()}")
                return
        except httpx.ReadTimeout:
            print(f"Request timed out during attempt {attempt + 1}. Retrying...")
            await asyncio.sleep(5)  # Wait before retrying
        except httpx.ConnectError:
            print(f"Connection attempt {attempt + 1} failed. Retrying...")
            await asyncio.sleep(2)

    raise RuntimeError("Failed to connect to the API after multiple attempts.")

if __name__ == "__main__":
    asyncio.run(test_process_image())