import numpy as np

NUM_BLOCKS = 8
NUM_DIMENSION = 3
VECTOR_LENGTH = 64

class TextWorld:
    def __init__(self, task=None):
        """
        We define the tabletop env as a coordinate of 8*1*8, where each block is a 1*1*1 cube.
        The coordinate system is right-handed, with the origin at the bottom-left corner of the tabletop.
        0 <= x <= 7, 0 <= y <= 0, 0 <= z <= 7
        and we eliminate the y-axis, so the coordinate is 8*8.
        0 <= x <= 7, 0 <= z <= 7
        The coordinate is a 2D numpy array, where each element is the index of the block.
        The number (name) of the block is 1-based, from 1 to 8.
        In the coordinate, the block is represented by its number (name).
        number == 0 means no block.
        """
        self.matrix_state = np.zeros((NUM_BLOCKS, NUM_BLOCKS+1), dtype=int)
        for i in range(NUM_BLOCKS):
            self.matrix_state[i][0] = i + 1
        self.vector = np.zeros(VECTOR_LENGTH, dtype=int)
        for i in range(56, 64):
            self.vector[i] = 1
        self.goal = task
    
    def get_state(self):
        return self.matrix_state
    
    def get_vector(self):
        return self.vector
    
    def ij2k(self, i, j):
        idx = -1
        for m in range(NUM_BLOCKS):
            for n in range(NUM_BLOCKS):
                if m == n:
                    continue
                idx += 1
                if m == i and n == j:
                    return idx
        assert False, "should not reach here"

    def k2ij(self, k):
        idx = -1
        for i in range(NUM_BLOCKS):
            for j in range(NUM_BLOCKS):
                if i == j:
                    continue
                idx += 1
                if idx == k:
                    return i, j
        assert False, "should not reach here"

    def query_column(self, obj):
        # obj is 1-based
        assert obj >= 1 and obj <= NUM_BLOCKS, "obj is out of range"
        x, y = np.where(self.matrix_state == obj)
        assert len(x) == 1 and len(y) == 1, "obj is not unique"
        x = x[0]
        y = y[0]
        below = []
        above = []
        for i in range(0, y):
            if i == y:
                continue
            if self.matrix_state[x][i] != 0:
                below.append(self.matrix_state[x][i])
        for i in range(y+1, NUM_BLOCKS+1):
            if self.matrix_state[x][i] != 0:
                above.append(self.matrix_state[x][i])
        return below, above
    
    def vector_to_natural_language(self, vector):
        natural_language = ""
        idx = -1
        for i in range(NUM_BLOCKS):
            for j in range(NUM_BLOCKS):
                if i == j:
                    continue
                idx += 1
                if vector[idx] == 1:
                    # print(f"On({i}, {j})")
                    natural_language += f"On({i}, {j}), "
        for k in range(56, 64):
            if vector[k] == 1:
                # print(f"Clear({k - 56})")
                natural_language += f"Clear({k - 55}), "
        natural_language += "\n"
        return natural_language

    def state_to_vector(self, matrix_state):
        vector = np.zeros(VECTOR_LENGTH, dtype=int)
        for i in range(NUM_BLOCKS):
            if matrix_state[i][0] == 0:
                continue
            for j in range(NUM_BLOCKS):
                if matrix_state[i][j] != 0 and matrix_state[i][j+1] != 0:
                    # On(obj_i, obj_j)
                    obj_i = matrix_state[i][j+1]
                    obj_j = matrix_state[i][j]
                    idx = self.ij2k(obj_i-1, obj_j-1)  # should be?
                    vector[idx] = 1
                elif matrix_state[i][j] != 0 and matrix_state[i][j+1] == 0:
                    # Clear(obj_i)
                    obj_i = matrix_state[i][j]
                    idx = 55 + obj_i
                    vector[idx] = 1
                    break
        return vector
    
    # def vector_to_state(self, vector):
    #     matrix_state = np.zeros((NUM_BLOCKS, NUM_BLOCKS+1), dtype=int)
    #     for i in range(56, 64):
    #         if vector[i] == 1:
    #             matrix_state[i-56][0] = i - 55
    #     idx = -1
    #     for i in range(VECTOR_LENGTH):
    #         for j in range(VECTOR_LENGTH):
    #             if i == j:
    #                 continue
    #             idx += 1
    #             if vector[idx] == 1:
    #                 obj_i = i
    #                 obj_j = j
    #                 matrix_state[obj_i][obj_j+1] = obj_j + 1
    #     return matrix_state
    
    def state_to_natual_language(self):
        natural_language = ""
        for i in range(NUM_BLOCKS):
            if self.matrix_state[i][0] == 0:
                continue
            for j in range(NUM_BLOCKS):
                if self.matrix_state[i][j] != 0 and self.matrix_state[i][j+1] != 0:
                    # On(obj_i, obj_j)
                    obj_i = self.matrix_state[i][j+1]
                    obj_j = self.matrix_state[i][j]
                    # print(f"On({obj_i}, {obj_j})")
                    natural_language += f"On({obj_i}, {obj_j}), "
                elif self.matrix_state[i][j] != 0 and self.matrix_state[i][j+1] == 0:
                    # Clear(obj_i)
                    obj_i = self.matrix_state[i][j]
                    # print(f"Clear({obj_i})")
                    natural_language += f"Clear({obj_i}), "
                    break
        natural_language += "\n"
        return natural_language
    
    def put_A_on_B(self, obj_A, obj_B):
        """
        In the current implementation, we ignore all the invalid actions.
        In the current implementation, invalid actions will make nothing happen, 
        without throwing an error. Nor will it give any warning.
        """
        # # put A on obj_B, such that On(A, B) holds
        # # obj_A and obj_B are index of blocks
        # # obj_A and obj_B are 1-based
        # # obj_A and obj_B are different
        # assert obj_A != obj_B, "obj_A and obj_B are the same"
        # # obj_A and obj_B are clear
        # _, above = self.query_column(obj_A)
        # assert above == [], "obj_A is not clear"
        # _, above = self.query_column(obj_B)
        # assert above == [], "obj_B is not clear"

        if obj_A == obj_B:
            return
        _, above = self.query_column(obj_A)
        if above != []:
            return
        _, above = self.query_column(obj_B)
        if above != []:
            return
        
        # move obj_A to the top of obj_B
        x_A, y_A = np.where(self.matrix_state == obj_A)
        assert len(x_A) == 1 and len(y_A) == 1, "obj A is not unique"
        x_A = x_A[0]
        y_A = y_A[0]
        x_B, y_B = np.where(self.matrix_state == obj_B)
        assert len(x_B) == 1 and len(y_B) == 1, "obj B is not unique"
        x_B = x_B[0]
        y_B = y_B[0]
        self.matrix_state[x_A][y_A] = 0
        self.matrix_state[x_B][y_B+1] = obj_A
        self.vector = self.state_to_vector(self.matrix_state)   # update vector
        return

    def step(self, action):
        # action is a string
        # action is in the form of a string like "put A on B"
        # A and B are 1-based
        # A and B are different
        assert action[0:4] == "put ", "action is not put"
        assert action[5:9] == " on ", "action is not on"
        obj_A = int(action[4])
        obj_B = int(action[9])
        self.put_A_on_B(obj_A, obj_B)
        # in the format of: obs, done, reward, info = env.step(action)
        obs = self.vector
        done = self.goal_reached()
        reward = 1 if done else 0
        info = ""
        return obs, self.goal, done, reward, info

    def feed_back(self):
        """
        [Optional] This may be used to give feedback when utilizing LLMs to generate actions.
        """
        # TODO
        raise NotImplementedError

    def set_task(self, task):
        # task in the form of a 64-bit vector
        self.goal = task
        return

    def reset(self, next_task=None):    
        self.matrix_state = np.zeros((NUM_BLOCKS, NUM_BLOCKS+1), dtype=int)
        for i in range(NUM_BLOCKS):
            self.matrix_state[i][0] = i + 1
        self.vector = np.zeros(VECTOR_LENGTH, dtype=int)
        for i in range(56, 64):
            self.vector[i] = 1
        self.set_task(next_task)
        # in the format of: obs, <goal>, done, reward, info = env.reset()
        obs = self.vector
        done = False
        reward = 0
        info = ""
        return obs, self.goal, done, reward, info
    
    def render(self):
        """
        We do not consider implementing a render with GUI currently.
        Only for debugging now.
        """
        print(self.matrix_state)
        print(self.vector)
        return
    
    def close(self):
        """
        [Optional] Close the environment.
        Simulators / environments like gymnasium usually use close() at the end of training.
        """
        # TODO
        raise NotImplementedError

    def goal_reached(self):
        """
        Check if self.vector is the same as self.goal
        """
        return np.array_equal(self.vector, self.goal)


if __name__ == "__main__":
    env = TextWorld()
    env.render()