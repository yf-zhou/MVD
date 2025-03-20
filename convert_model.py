import torch 

dirname = "/home/medcvr/yifei/thesis/runs/dvrk_reach_mvd/3191315/trained_models/env_step1000/"
filename = "models.pt"

models_dict = torch.load(dirname + filename)

from algorithms import make_agent
from unity_config import reach_cfg

obs_shape = (9, 84, 84)
action_shape = (2,)
action_range = [-1.0, 1.0]

agent = make_agent(obs_shape, action_shape, action_range, reach_cfg, None)

agent.actor.load_state_dict(models_dict["actor"])
# agent.critic.load_state_dict(models_dict["critic"])
# agent.critic_target.load_state_dict(models_dict["critic_target"])

example_obs = (torch.randn(1, 9, 84, 84).to(device=reach_cfg.device), None, False, False, False, True)

onnx_actor = torch.onnx.export(agent.actor, example_obs, dynamo=True)
onnx_actor.optimize()

onnx_actor.save(dirname + "actor.onnx")

# check onnx
import onnx

onnx_actor = onnx.load(dirname + "actor.onnx")

onnx.checker.check_model(onnx_actor)

# test
import onnxruntime

onnx_inputs = [tensor.numpy(force=True) if isinstance(tensor, torch.Tensor) else tensor for tensor in example_obs]

ort_session = onnxruntime.InferenceSession(
    dirname + "actor.onnx",
    providers=["CPUExecutionProvider"]
)

onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]

print(onnxruntime_outputs)

print(agent)