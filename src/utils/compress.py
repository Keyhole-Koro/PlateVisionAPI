import cv2

def compress_image(image, target_size=(800, 600), compression_quality=80):
    """
    Resize the image to a target size while maintaining the aspect ratio.
    Compression quality controls the JPEG file size and quality (0-100).
    """
    # Step 1: Resize the image to maintain the aspect ratio
    h, w = image.shape[:2]
    aspect_ratio = w / h
    
    new_w = target_size[0]
    new_h = int(new_w / aspect_ratio)
    
    if new_h > target_size[1]:
        new_h = target_size[1]
        new_w = int(new_h * aspect_ratio)
    
    resized_image = cv2.resize(image, (new_w, new_h))

    # Step 2: Compress the image using JPEG format and compression quality
    # Use imencode to compress to JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality]
    result, compressed_image = cv2.imencode('.jpg', resized_image, encode_param)
    
    if not result:
        raise ValueError("Error in compressing image")
    
    # Decode back to image format (it was compressed to byte stream)
    compressed_image = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)

    return compressed_image
