import os
import cv2
import retro
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.policies import ActorCriticCnnPolicyImproved
from stable_baselines3.common.atari_wrappers import WarpFrame
from components import StochasticFrameSkip, ScoreRewardEnv


BUTTON_LABELS = ['B', 'NULL', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT']


PANEL_WIDTH = 195
UPSCALE = 3  

BTN_SIZE = 50  
MARGIN = 20    


KEY_LAYOUT = {
    'UP':    (PANEL_WIDTH//2 - BTN_SIZE//2, MARGIN),
    'LEFT':  (MARGIN, PANEL_WIDTH//2 - BTN_SIZE//2),
    'DOWN':  (PANEL_WIDTH//2 - BTN_SIZE//2, PANEL_WIDTH - BTN_SIZE - MARGIN),
    'RIGHT': (PANEL_WIDTH - BTN_SIZE - MARGIN, PANEL_WIDTH//2 - BTN_SIZE//2),
    'B':     (PANEL_WIDTH//2 - BTN_SIZE//2, PANEL_WIDTH + MARGIN)
}

BTN_BG      = (50,  50,  50)  
BTN_ACTIVE  = (0, 200,   0)   
TEXT_COLOR  = (255,255,255)  

def draw_overlay(frame_rgb, action_bin):

    h, w = frame_rgb.shape[:2]
    game_up = cv2.resize(frame_rgb, (w*UPSCALE, h*UPSCALE), interpolation=cv2.INTER_NEAREST)
    h_up, w_up = game_up.shape[:2]

    panel_h = max(h_up, PANEL_WIDTH + BTN_SIZE + 2*MARGIN)
    panel = np.zeros((panel_h, PANEL_WIDTH, 3), dtype=np.uint8)

    for label, (x, y) in KEY_LAYOUT.items():

        idx = BUTTON_LABELS.index(label)
        pressed = bool(action_bin[idx])

        tl = (x, y)
        br = (x + BTN_SIZE, y + BTN_SIZE)
        color = BTN_ACTIVE if pressed else BTN_BG
        cv2.rectangle(panel, tl, br, color, thickness=-1, lineType=cv2.LINE_AA)

        cx, cy = x + BTN_SIZE//2, y + BTN_SIZE//2
        if label in ('UP','DOWN','LEFT','RIGHT'):
            if label == 'UP':
                pts = np.array([[cx, cy-15],[cx-10, cy+10],[cx+10, cy+10]], np.int32)
            elif label == 'DOWN':
                pts = np.array([[cx, cy+15],[cx-10, cy-10],[cx+10, cy-10]], np.int32)
            elif label == 'LEFT':
                pts = np.array([[cx-15, cy],[cx+10, cy-10],[cx+10, cy+10]], np.int32)
            else:  
                pts = np.array([[cx+15, cy],[cx-10, cy-10],[cx-10, cy+10]], np.int32)
            cv2.fillPoly(panel, [pts], TEXT_COLOR)
        else:
        
            cv2.putText(panel, 'B', (x + 12, y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEXT_COLOR, 2, cv2.LINE_AA)

    if panel_h > h_up:
        pad_top = (panel_h - h_up) // 2
        pad_bot = panel_h - h_up - pad_top
        game_up = np.vstack([
            np.zeros((pad_top,  w_up, 3), dtype=np.uint8),
            game_up,
            np.zeros((pad_bot,  w_up, 3), dtype=np.uint8)
        ])

    combined = np.hstack([game_up, panel])
    return combined


def wrap_env(env):
    env = StochasticFrameSkip(env, n=4, stickprob=0.25, SEED=57)
    env = WarpFrame(env)
    env = ScoreRewardEnv(env)
    return env

def make_env(game, state):
    def _init():
        env = retro.make(game=game, state=state or retro.State.DEFAULT, render_mode="rgb_array")
        return wrap_env(env)
    return _init

CUSTOM_OBJECTS = { "policy_class": ActorCriticCnnPolicyImproved}


def record_video(model_path,
                 game="Riverraid-Atari2600",
                 state=None,
                 video_path="output.mp4",
                 max_steps=10000):

    env = DummyVecEnv([make_env(game, state)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    model = PPO.load(model_path, custom_objects=CUSTOM_OBJECTS)

    obs = env.reset()
    frame = env.envs[0].render()  

    dummy_action = np.zeros((1, len(BUTTON_LABELS)), dtype=int)
    combined = draw_overlay(frame, dummy_action[0])
    h0, w0 = combined.shape[:2]

    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,         
        (w0, h0)      
    )

    total_reward = 0.0
    done = False
    steps = 0


    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        result = env.step(action)

        if len(result) == 4:
            obs, reward, terminated, info = result
            truncated = terminated
        else:
            obs, reward, terminated, truncated, info = result

        done = terminated[0] or truncated[0]
        total_reward += reward[0]

        frame = env.envs[0].render()       
        action_bin = action[0] if isinstance(action[0], np.ndarray) else np.array(action[0])
        combined = draw_overlay(frame, action_bin)

        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        steps += 1

    out.release()
    env.close()
    print(f"\nVideo saved: {video_path}")
    print(f"Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="selected_model")
    parser.add_argument("--game",      default="Riverraid-Atari2600")
    parser.add_argument("--state",     default=None)
    parser.add_argument("--output",    default="evaluation_video.mp4")
    parser.add_argument("--max_steps", type=int, default=10000)
    args = parser.parse_args()

    record_video(
        model_path=args.model,
        game=args.game,
        state=args.state,
        video_path=args.output,
        max_steps=args.max_steps
    )
