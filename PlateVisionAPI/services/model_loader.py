from loaders.initialLoad import initial_load_models

async def load_initial_models():
    """Load all models asynchronously."""
    (
        model_splitting_sections,
        model_LicensePlateDet,
        classification_model,
        classification_scaler,
    ) = await initial_load_models()

    return {
        "model_splitting_sections": model_splitting_sections,
        "model_LicensePlateDet": model_LicensePlateDet,
        "classification_model": classification_model,
        "classification_scaler": classification_scaler,
    }
