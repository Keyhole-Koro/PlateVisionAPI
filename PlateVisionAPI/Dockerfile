FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

# Copy requirements and install only CPU-related packages
COPY requirements.txt .

RUN yum install -y mesa-libGL

RUN pip install --upgrade pip \
    && pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install albumentations==1.4.20 \
    && pip install --no-cache-dir $(grep -vE "^(#|$)" requirements.txt | grep -ivE "tensorflow-gpu|cupy|torch.*cuda|nvidia") \
    && rm -rf /root/.cache/pip \
    && pip uninstall -y pip setuptools wheel \
    && rm -rf /usr/local/lib/python3.11/site-packages/pip* \
    /usr/local/lib/python3.11/site-packages/setuptools* \
    /usr/local/lib/python3.11/site-packages/wheel* \
    /usr/local/bin/pip* /usr/local/bin/easy_install* /usr/local/bin/wheel*

# Copy application code
COPY . .

# Set the Lambda handler
CMD ["lambda_handler.handler"]
