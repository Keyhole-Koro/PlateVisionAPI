from mangum import Mangum
from app.main import app

import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"
os.environ["MPLCONFIGDIR"] = "/tmp"

handler = Mangum(app)
