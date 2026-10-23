import os; os.environ["MS_SKIP_ASSET_DOWNLOAD_PROMPT"]="1"
import gymnasium as gym, numpy as np, imageio, mani_skill.envs
ids = ["RotateSingleObjectInHandLevel0-v1","RotateSingleObjectInHandLevel2-v1"]
for i in ids:
    env = gym.make(i, num_envs=1, render_mode="rgb_array", sim_backend="gpu")
    env.reset(seed=0)
    frames = []
    for t in range(40):
        env.step(env.action_space.sample() * 0.3)
        frames.append(env.render().cpu().numpy()[0].astype(np.uint8))
    imageio.imwrite(f"{i}.png", frames[-1])
    imageio.mimsave(f"{i}.mp4", frames, fps=20)
    env.close(); print("done", i, flush=True)
