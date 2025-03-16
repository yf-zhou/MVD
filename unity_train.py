import time
from unity_workspace import Workspace, default_cfg

def main(cfg):
    global workspace
    workspace = Workspace(cfg)
    start_time = time.time()
    workspace.run()
    print(f"total run time: {time.time()-start_time}")

if __name__ == '__main__':
    # cfg = parse_args()
    cfg = default_cfg
    main(cfg)