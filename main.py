from TextWorld import TextWorld
from CP_sparse import continous_planner, clip
import numpy as np
import torch
import torch.nn.functional as F
from encoder import Encoder, MLP, Deep_MLP
from tqdm import tqdm
import argparse
import pickle

NUM_BLOCKS = 8
VECTOR_LENGTH = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--clip", type=int, default=0, help="if clip")
parser.add_argument("--continuous", type=int, default=0, help="if continuous")
args = parser.parse_args()
    

if __name__ == "__main__":
    env = TextWorld()
    tasks = np.load("./dataset/tasks.npy")
    with open("./dataset/goal_set_2000.pkl", "rb") as f:
        goal_set = pickle.load(f)
    goal_list = goal_set["data"]

    for epoch in range(1, 51):
        print(f"epoch {epoch}")
        encoder = Encoder(NUM_BLOCKS*2, 128, VECTOR_LENGTH).to(device)
        if args.continuous:
            state_dict = torch.load(f"./model/all_1900/OSIL_state_dict_{epoch}.pth", map_location="cpu")
            encoder.load_state_dict(state_dict)
            
        success_cnt = 0
        for episode in tqdm(range(1900, 2000)):
            # print(f"task {episode}")
            env.set_task(tasks[episode])
            obs, goal, done, reward, info = env.reset(next_task=tasks[episode])
            step = 0
            init_description = env.state_to_natual_language()
            embedding_goal = torch.Tensor(goal_list[episode]).reshape(1, -1).to(device)
            embedding_goal = F.sigmoid(encoder(embedding_goal))[0].detach().cpu().numpy()   # in python 3.9, we should add "[0]". In python 3.10, we should not add "[0]".
            while not done and step < 2:
                step += 1
                if args.continuous:
                    coordinate = env.state_to_coordinate()
                    coordinate = torch.Tensor(coordinate).reshape(1, -1).to(device)
                    embedding_vector = encoder(coordinate)
                    embedding_vector = F.sigmoid(embedding_vector)[0].detach().cpu().numpy()    # in python 3.9, we should add "[0]". In python 3.10, we should not add "[0]".
                    actions = continous_planner(embedding_vector, embedding_goal, if_clip=args.clip)
                else:
                    actions = continous_planner(obs, goal, if_clip=args.clip)
                for action in actions:
                    next_obs, goal, done, reward, info = env.step(action)
                    current_state_description = env.state_to_natual_language()
                    obs = next_obs
                    assert obs.sum() == 8, f"obs.sum()"
            if done:
                success_cnt += 1
        print(f"success rate: {success_cnt} / 100")
        
