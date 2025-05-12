https://github.com/PaddlePaddle/Paddle2ONNX

paddle2onnx --model_dir ../PlateVisionAPI/models/paddle/number \
            --model_filename inference.json \
            --params_filename inference.pdiparams \
            --save_file inference.onnx

paddle2onnx --model_dir models\paddle\classification --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file inference.onnx
