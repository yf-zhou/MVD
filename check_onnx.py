import onnxruntime
import matplotlib.pyplot as plt 
import numpy as np

dirname = "/home/medcvr/yifei/thesis/output/"
files = [f"unity_obs_{i}.png" for i in range(10)]

modelname = "/home/medcvr/yifei/thesis/runs/dvrk_reach_mvd/3201550/trained_models/env_step15000/actor.onnx"

ort_session = onnxruntime.InferenceSession(
    modelname,
    providers=["CPUExecutionProvider"]
)

for file in files:
    im = plt.imread(dirname+file)
    im = np.expand_dims(np.moveaxis(im, -1, 0), axis=0)
    onnx_input = [im, None, False, False, True, True]
    
    onnx_in = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_input)}
    onnx_out = ort_session.run(None, onnx_in)[0]

    print(onnx_out)