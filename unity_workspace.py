import gymnasium as gym
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

import os
import time
import logging
from logger import Logger
import collections
import numpy as np
import torch

import unity_utils as utils
import algorithms 
from algorithms.replay_buffer import ReplayBuffer
from video import VideoRecorder

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg

        # current working directory + check if unique
        self.work_dir = os.path.join(os.getcwd(), cfg.log_dir, cfg.exp_name, str(cfg.seed))
        assert not os.path.exists(self.work_dir), f'specified working directory {self.work_dir} already exists' 
        os.makedirs(self.work_dir)       

        # logging
        logger = logging.getLogger(__name__)
        logger.setLevel("DEBUG")

        # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # file_handler = logging.FileHandler(f"workspace_{timestamp}.log", encoding='utf-8')
        file_handler = logging.FileHandler(f"workspace_{cfg.seed}.log", mode='a', encoding='utf-8')
        formatter = logging.Formatter(
            "[{asctime} - {levelname}]: {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel("DEBUG")
        logger.addHandler(file_handler)
        self.logger = logger

        self.logger2 = Logger(self.work_dir,
                             log_frequency=self.cfg.log_freq,
                             action_repeat=self.cfg.action_repeat,
                             eval_on_each_scenario=(self.cfg.eval_on_each_camera or len(self.cfg.cameras)==1),
                             domain_name=self.cfg.domain_name)
        
        print(f"workspace: {self.work_dir}")
        self.logger.info(f"{self.work_dir=}")
        self.save_dir = os.path.join(self.work_dir, "trained_models")

        # setup
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(seed=cfg.seed)

        self.cameras = cfg.cameras

        # environments
        self.unity_env = UnityEnvironment(cfg.build_executable)
        self.env = UnityToGymWrapper(self.unity_env, uint8_visual=True, allow_multiple_obs=True, action_space_seed=cfg.seed)
        self.unity_eval_env = UnityEnvironment(cfg.build_executable, worker_id=1)
        self.eval_env = UnityToGymWrapper(self.unity_eval_env, uint8_visual=True, allow_multiple_obs=True, action_space_seed=cfg.seed+100)

        self.env.reset()
        self.eval_env.reset()

        # agents and replay buffer
        action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        # obs_shape = (len(self.env.observation_space), *self.env.observation_space[0].shape)
        obs_shape = (len(self.env.observation_space) * len(self.cameras), *self.env.observation_space[0].shape[1:])
        print(obs_shape, type(obs_shape))
        print(self.env.action_space.shape, type(self.env.action_space.shape))
        print(action_range, type(action_range))
        self.agent = algorithms.make_agent(obs_shape, self.env.action_space.shape, action_range, cfg, None)

        self.replay_buffer = ReplayBuffer(
            # self.env.observation_space.shape,
            obs_shape,
            self.env.action_space.shape,
            cfg.replay_buffer_capacity,
            cfg.image_pad,
            self.device,
            None
        )

        # videos
        self.video_dir = os.path.join(self.work_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        self.video_recorder = VideoRecorder()
        
        self.step = 0

    def evaluate(self, record_video=False):
        average_episode_reward = 0
        successes = 0

        if record_video:
            self.video_recorder.init(enabled=True)

        for episode in range(self.cfg.num_eval_episodes):
            obs = np.vstack(self.eval_env.reset())

            done = False
            episode_reward = 0
            episode_step = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, None, sample=False)

                obs, reward, terminated, info = self.eval_env.step(action)
                obs = np.vstack(obs)
                done = terminated

                if record_video:
                    self.video_recorder.record(self.eval_env)

                episode_reward += reward
                episode_step += 1

            average_episode_reward += episode_reward
            # TODO: look into tracking successes
            # try:
            #     successes += info['success']
            # except:
            #     successes += info['is_success']

            if record_video:
                self.video_recorder.save(os.path.join(self.video_dir, f'eval_{episode}.mp4'))
                self.video_recorder.reset()

        # TODO: not sure about this logging
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger2.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.info(f"step: {self.step:05}\teval/episode_reward: {average_episode_reward}")

        # success_rate = successes / self.cfg.num_eval_episodes
        # self.logger2.log('eval/success_rate', success_rate, self.step)
        self.logger2.dump(self.step, save=True, ty="eval")

        if len(self.cameras) == 1:
            self.logger2.log(f'eval_scenarios/{self.cameras[0]}_cam_episode_reward', average_episode_reward, self.step)
            # self.logger.log(f'eval_scenarios/{self.cameras[0]}_cam_success_rate', success_rate, self.step)
            self.logger2.dump(self.step, save=True, ty="eval_scenarios")

    def evaluate_on_each_scenario(self, record_video=False):
        for idx, cam in enumerate(self.cameras):
            average_episode_reward = 0
            successes = 0

            if record_video:
                self.video_recorder.init(enabled=True)

            for episode in range(self.cfg.num_eval_episodes):
                obs = np.vstack(self.eval_env.reset())
                obs = obs.reshape(len(self.cameras), -1, *obs.shape[1:])[idx]

                done = False
                episode_reward = 0
                episode_step = 0

                while not done:
                    with utils.eval_mode(self.agent):
                        action = self.agent.act(obs, None, sample=False, eval_on_single_cam=True)

                    obs, reward, terminated, info = self.eval_env.step(action)
                    obs = np.vstack(obs)
                    obs = obs.reshape(len(self.cameras), -1, *obs.shape[1:])[idx]

                    done = terminated
                    if record_video:
                        self.video_recorder.record(self.eval_env, camera=cam)
                    episode_reward += reward
                    episode_step += 1

                average_episode_reward += episode_reward
                # TODO: fix
                # try:
                #     successes += info['success']
                # except:
                #     successes += info['is_success']
                if record_video:
                    self.video_recorder.save(os.path.join(self.video_dir, f'eval_scenarios_{cam}_cam_{episode}.mp4'))
                    self.video_recorder.reset()

            average_episode_reward /= self.cfg.num_eval_episodes
            success_rate = successes / self.cfg.num_eval_episodes

            self.logger2.log(f'eval_scenarios/{cam}_cam_episode_reward', average_episode_reward, self.step)
            self.logger2.log(f'eval_scenarios/{cam}_cam_success_rate', success_rate, self.step)

        self.logger2.dump(self.step, save=True, ty="eval_scenarios")

    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()

        total_num_steps = self.cfg.num_train_steps

        successes = collections.deque([], maxlen=10)

        while self.step <= (total_num_steps + 1):
            if done:
                if self.step>0:
                    # try:
                    #     successes.append(info['success'])
                    # except:
                    #     successes.append(info['is_success'])

                    self.logger2.log('train/episode_reward', episode_reward, self.step)
                    # self.logger.log('train/success_rate', np.mean(successes), self.step)
                    self.logger2.log('train/duration', time.time() - start_time, self.step)
                    self.logger.info(f"step: {self.step:05}\ttrain/episode_reward: {episode_reward}")
                    self.logger.info(f"step: {self.step:05}\ttrain/duration: {time.time() - start_time}")
                    
                    start_time = time.time()
                    self.logger2.dump(self.step, save=(self.step > self.cfg.num_seed_steps), ty="train")

                obs = np.vstack(self.env.reset())

                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger2.log('train/episode', episode, self.step)
                self.logger.info(f"step: {self.step:05}\ttrain/episode: {episode}")

            # evaluate agent periodically
            if self.step % self.cfg.eval_freq == 1:
                self.logger2.log('eval/episode', episode, self.step)
                self.logger.info(f"step: {self.step:05}\teval/episode: {episode}")
                self.evaluate(record_video=self.cfg.save_video)
                if self.cfg.eval_on_each_camera:
                    self.logger2.log('eval_scenarios/episode', episode, self.step)
                    self.logger.info(f"step: {self.step:05}\teval_scenarios/episode: {episode}")
                    self.evaluate_on_each_scenario(record_video=self.cfg.save_video)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, proprioceptive_state=info.get("proprioceptive_state"), sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger2, self.step)

            if self.step > 0 and self.step % self.cfg.save_freq == 0:
                saveables = {
                    "actor": self.agent.actor.state_dict(),
                    "critic": self.agent.critic.state_dict(),
                    "critic_target": self.agent.critic_target.state_dict()
                }
                save_at = os.path.join(self.save_dir, f"env_step{self.step * self.cfg.action_repeat}")
                os.makedirs(save_at, exist_ok=True)
                torch.save(saveables, os.path.join(save_at, "models.pt"))

            next_obs, reward, terminated, info = self.env.step(action)
            next_obs = np.vstack(next_obs)
            # print(next_obs)
            # print(reward)
            # print(terminated)
            # print(truncated)
            # print(info)


            # allow infinite bootstrap
            done = terminated
            # done_no_max = 0 if truncated else done
            done_no_max = done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max, episode, None)

            obs = next_obs
            episode_step += 1
            self.step += 1