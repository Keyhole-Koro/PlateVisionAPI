import asyncio
import httpx
import os
import time
import base64

API_URL = "https://n2oflso7fky7tjztxcrcss3b2u0noyjb.lambda-url.ap-northeast-3.on.aws/process_image"  # Use the correct address
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
            async with httpx.AsyncClient(timeout=httpx.Timeout(50.0)) as client:  # Set timeout to 50 seconds
                start_time = time.time()
                async with httpx.AsyncClient(verify=False) as client:
                    response = await client.post(
                        API_URL,
                        json=json_payload,
                        params={"measure": "true", "return_image_annotation": "false"},
                        headers={"Authorization": f"Bearer {API_KEY}"}
                    )
                elapsed_time = time.time() - start_time
                assert response.status_code == 200, f"Unexpected status code: {response.status_code}...{response}"
                print(f"Test passed. Response: {response.json()}, Time taken: {elapsed_time:.2f} seconds")
                return
        except httpx.ConnectTimeout:
            print(f"Request timeout during attempt {attempt + 1}. Retrying...")
            await asyncio.sleep(5)  # Increase sleep time between retries
        except httpx.ConnectError:
            print(f"Connection attempt {attempt + 1} failed. Retrying...")
            await asyncio.sleep(2)

    raise RuntimeError("Failed to connect to the API after multiple attempts.")

if __name__ == "__main__":
    asyncio.run(test_process_image())