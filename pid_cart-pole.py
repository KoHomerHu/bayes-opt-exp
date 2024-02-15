import gymnasium as gym

class PID:
    def __init__(self, kp, kd, ki, goal):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.goal = goal
        self.prev_error = 0
        self.integral = 0

    def control(self, state):
        error = self.goal - state
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.kd * derivative + self.ki * self.integral


class Sampler:
    def __init__(self, num_episodes, max_steps, env):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.env = env

    def sample(self, parameters):
        if len(parameters) == 6 or len(parameters) == 4:
            if len(parameters) == 6:
                kp1, kd1, ki1, kp2, kd2, ki2 = parameters
                controller_angle = PID(kp1, kd1, ki1, goal=0)
                controller_position = PID(kp2, kd2, ki2, goal=0)
            else:
                kp1, kd1, kp2, kd2 = parameters
                controller_angle = PID(kp1, kd1, 0, goal=0)
                controller_position = PID(kp2, kd2, 0, goal=0)

            ret = 0
            ret_lst = []

            for _ in range(self.num_episodes):
                observation, _ = env.reset()
                for _ in range(self.max_steps):
                    pole_angle = observation[2]
                    control_output_angle = controller_angle.control(pole_angle)
                    control_output_position = controller_position.control(observation[0])
                    action = 1 if control_output_angle + control_output_position < 0 else 0

                    observation, reward, terminated, truncated, info = env.step(action)
                    ret += reward
                    if terminated or truncated:
                        break
                ret_lst.append(ret)
                ret = 0
        else:
            assert len(parameters) == 3
            kp, kd, ki = parameters
            controller = PID(kp, kd, ki, goal=0)
            
            ret = 0
            ret_lst = []

            for _ in range(self.num_episodes):
                observation, _ = env.reset()
                for _ in range(self.max_steps):
                    pole_angle = observation[2]
                    control_output = controller.control(pole_angle)
                    action = 1 if control_output < 0 else 0

                    observation, reward, terminated, truncated, info = env.step(action)
                    ret += reward
                    if terminated or truncated:
                        break
                ret_lst.append(ret)
                ret = 0

        return sum(ret_lst) / len(ret_lst)
    
if __name__ == "__main__":
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.acquisition import UpperConfidenceBound
    from botorch.optim import optimize_acqf
    import torch
    import warnings
    warnings.filterwarnings("ignore")

    n = 5 # initial number of data points
    num_episodes = 50
    max_steps = 100
    env = gym.make("CartPole-v1", render_mode="human")
    sampler = Sampler(num_episodes, max_steps, env)
    train_X = 1000 * torch.rand(n, 4)
    train_Y = torch.zeros(n, 1)
    for i in range(n):
        parameters = train_X[i].tolist()
        train_Y[i] = sampler.sample(parameters)
    train_Y = (train_Y - train_Y.mean()) / train_Y.std()

    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    af = UpperConfidenceBound(gp, beta=0.5)

    bounds = torch.tensor([[
        1e-5, 1e-5, 1e-5, 1e-5
    ], [
        1.0, 1.0, 1.0, 1.0
    ]])

    best_ret = 0
    best_parameters = None
    ret_lst = []

    for _ in range(100):
        if len(ret_lst) > 1:
            if abs(ret_lst[-1] - ret_lst[-2]) < 1e-5:
                break
        candidate, acq_value = optimize_acqf(
            acq_function=af,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        train_X = torch.cat([train_X, candidate], dim=0)
        parameters = (1000 * candidate).tolist()[0]
        avg_ret = sampler.sample(parameters)
        train_Y = torch.cat([train_Y, torch.tensor([[avg_ret,],])], dim=0)
        train_Y = (train_Y - train_Y.mean()) / train_Y.std()
        gp = SingleTaskGP(train_X, train_Y)
        try:
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
        except:
            print("Error fitting the model")
            pass
        af = UpperConfidenceBound(gp, beta=2.5)
        # af = ExpectedImprovement(gp, best_f=train_Y.max())

        ret_lst.append(avg_ret)

        if avg_ret > best_ret:
            best_ret = avg_ret
            best_parameters = parameters

        kp1, kd1, kp2, kd2 = parameters
        print("For parameters ({:.2f}, {:.2f}, {:.2f}, {:.2f}), the average return is {}".format(kp1, kd1, kp2, kd2, avg_ret))

    env.close()

    kp1, kd1, kp2, kd2 = best_parameters
    print("Best parameters are ({:.2f}, {:.2f}, {:.2f}, {:.2f}) with average return {}".format(kp1, kd1, kp2, kd2, avg_ret))

    import matplotlib.pyplot as plt
    plt.plot(ret_lst)
    plt.title("Average return over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Average return")
    plt.show()
    

