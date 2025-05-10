import asyncio
from services.classification.classify_plate import safe_load_joblib

async def load_classifying_models(classifying_model):
    loop = asyncio.get_event_loop()
    model = await loop.run_in_executor(None, safe_load_joblib, classifying_model["model"]["path"])
    scaler = await loop.run_in_executor(None, safe_load_joblib, classifying_model["scaler"]["path"])
    return {
        "model": model,
        "scaler": scaler,
    }

async def load_detection_models(detection_config):
    tasks = []
    for config in detection_config.values():
        instance = config["engine"](config["path"])
        config["engine_instance"] = instance
        tasks.append(instance.load_model())
    await asyncio.gather(*tasks)

async def load_ocr_models(ocr_config):
    tasks = []
    for config in ocr_config.values():
        if config is None or config.get("engine") is None:
            continue
        instance = config["engine"](config["path"], config["dict_path"])
        config["engine_instance"] = instance
        tasks.append(instance.load_model())
    await asyncio.gather(*tasks)

async def load_all_models(classifying_model, detection_config, ocr_config):
    classification_task = load_classifying_models(classifying_model)
    detection_task = load_detection_models(detection_config)
    ocr_task = load_ocr_models(ocr_config)

    classification, _, _ = await asyncio.gather(classification_task, detection_task, ocr_task)

    return {
        "classification": classification,
        "detection": detection_config,
        "ocr": ocr_config,
    }
