import numpy as np

# torch imports
import torch

import morel.models.utils as utils
import os
class FakeEnv:
    def __init__(self, dynamics_model, mdl, s_swag,
                        obs_mean,
                        obs_std,
                        action_mean,
                        action_std,
                        delta_mean,
                        delta_std,
                        reward_mean,
                        reward_std,
                        initial_obs_mean,
                        initial_obs_std,
                        start_states,
                        timeout_steps = 300,
                        uncertain_penalty = -100,
                        device = "cuda:0",
                        ):
        self.dynamics_model = dynamics_model
        
        #print(f'weight {sum(p.numel() for p in self.dynamics_model.parameters())}')

        self.uncertain_penalty = uncertain_penalty
        self.start_states = start_states

        self.input_dim = self.dynamics_model.input_dim
        self.output_dim = self.dynamics_model.output_dim

        self.device = device

        # Save data transform parameters
        self.obs_mean = torch.Tensor(obs_mean).float().to(self.device)
        self.obs_std = torch.Tensor(obs_std).float().to(self.device)
        self.action_mean = torch.Tensor(action_mean).float().to(self.device)
        self.action_std = torch.Tensor(action_std).float().to(self.device)
        self.delta_mean = torch.Tensor(delta_mean).float().to(self.device)
        self.delta_std = torch.Tensor(delta_std).float().to(self.device)
        self.reward_mean = torch.Tensor([reward_mean]).float().to(self.device)
        self.reward_std = torch.Tensor([reward_std]).float().to(self.device)

        self.initial_obs_mean = torch.Tensor(initial_obs_mean).float().to(self.device)
        self.initial_obs_std = torch.Tensor(initial_obs_std).float().to(self.device)

        self.timeout_steps = timeout_steps

        self.state = None
        self.mdl = mdl
        self.s_swag = s_swag

        # swag related variables
        if self.mdl == 'swag':

            self.param_dict = self.dynamics_model.swag()
            self.K = self.param_dict["K"]
            self.var = torch.clamp(self.param_dict["sigma_diag"], 1e-30).cuda()
            self.sqrt_var = self.var.sqrt()
            self.cov = torch.tensor(self.param_dict["D"], dtype=torch.float32).cuda()
            self.inv_sqrt_Kminus1 = (1 / torch.sqrt(torch.tensor(self.K-1)))

    def reset(self):
        idx = np.random.choice(self.start_states.shape[0])
        next_obs = torch.tensor(self.start_states[idx]).float().to(self.device)
        self.state = (next_obs - self.obs_mean)/self.obs_std
        # print("reset!")
        # self.state = torch.normal(self.obs_mean, self.obs_std)
        # next_obs = self.obs_mean + self.obs_std*self.state
        self.steps_elapsed = 0

        return next_obs

    def step(self, action_unnormalized, obs = None):
        action = (action_unnormalized - self.action_mean)/self.action_std

        if obs is not None:
            self.state = (torch.tensor(obs).float().to(self.device) - self.obs_mean)/self.obs_std
            # self.state = torch.unsqueeze(self.state,0)
            print(self.state.shape)
            print(action.shape)
        
        if self.mdl == 'ensemble':
            predictions = self.dynamics_model.predict(torch.cat([self.state, action],0))
        
        elif self.mdl == 'swag':
            pred_list = []      
            theta_list = []
            sampled_theta_list = []
            for s in range(self.s_swag):
                #print(f'K is {K}')
                #print(f'D is {param_dict["D"]}')            
                #print(f'D type is {type(param_dict["D"])}')

                # Draw diagonal variance sample
                z1 = torch.randn_like(self.var, requires_grad=False)
                var_sample = self.sqrt_var * z1 # tensor
                # var_sample = var_sample.cuda() XXX check var_sample is in cuda

                # Draw low rank sample 
                #cov_sample = (1 / np.sqrt(2 * (K - 1))) * (param_dict["D"] @ np.random.normal(np.zeros((K,)), np.ones((K,)))) # numpy
                z2 = torch.randn((self.K,1)).cuda()
                cov_sample = self.inv_sqrt_Kminus1 * self.cov.matmul(z2) # tensor # size (p,1)
                cov_sample = torch.flatten(cov_sample, 0) # size (p)
                #print(f'cov  {cov_sample}')
                #print(f'var  {var_sample}')



                rand_sample = var_sample+cov_sample #tensor # size (p)
                theta_list.append(rand_sample[10])
        
                sample = self.param_dict['theta_swa']+1/np.sqrt(2)*rand_sample #tensor # size (p)
                sampled_theta_list.append(sample[10])
                #print(sample.size())
                assert type(sample) is torch.Tensor

                #sample = torch.unsqueeze(sample, 0) # size (1, p)
                #print(sample.size())

                sample_dict = utils.unflatten_like_dict(sample, self.dynamics_model.named_parameters())
                
                state_dict = self.dynamics_model.state_dict()
                for name, _ in self.dynamics_model.named_parameters():
                    state_dict[name] = sample_dict[name]
                self.dynamics_model.load_state_dict(state_dict)
                # pred = self.dynamics_model.predict(torch.cat([self.state, action],0)).detach().cpu()
                pred = self.dynamics_model.predict(torch.cat([self.state, action],0))
                #predictions = np.append(predictions, pred)
                # predictions = np.append(predictions, pred.reshape(1, self.output_dim),0)
                pred_list.append(pred)
                #print(f'first predictions {predictions}')
                #torch.save(self.dynamics_model.state_dict(), os.path.join('result/', f'dynamics{s}.pt'))

    
            #predictions = torch.tensor(predictions, device = 'cuda').float()
            predictions = torch.stack(pred_list, axis=0).float()
            #predictions = self.dynamics_model.predict(torch.cat([self.state, action],0))
            #print(f'predictions {predictions}')
            #print(f'theta list {theta_list}')
            #print(f'sampled theta list {sampled_theta_list}')

        deltas = predictions[:,0:self.output_dim-1]

        rewards = predictions[:,-1]

        #predictions : tensor, shape (S, 5), cuda
        #delta, state, next_obs, self.state : tensor, shape (4), cuda
        #uncertain : True or False
        #reaward_out : float

        # Calculate next state
        deltas_unnormalized = self.delta_std*torch.mean(deltas,0).cuda() + self.delta_mean
        state_unnormalized = self.obs_std*self.state + self.obs_mean
        next_obs = deltas_unnormalized + state_unnormalized
        self.state = (next_obs - self.obs_mean)/self.obs_std

        uncertain = self.dynamics_model.usad(predictions.cpu().numpy())

        reward_out = self.reward_std*torch.mean(rewards) + self.reward_mean

        if(uncertain):
            reward_out[0] = self.uncertain_penalty
        reward_out = torch.squeeze(reward_out)

        self.steps_elapsed += 1
        # print("reward {}\tuncertain{}".format(reward_out, uncertain))
        # input()
        #print(f'deltas_unnormalized {deltas_unnormalized}')
        #print(f'state_unnormalized {state_unnormalized}')
        #print(f'next_obs {next_obs}')
        #print(f'self.state {self.state}')
        #print(f'uncertain {uncertain}')
        #print(f'reward_out {reward_out}')
        return next_obs, reward_out, (uncertain or self.steps_elapsed > self.timeout_steps), {"HALT" : uncertain}

        '''
        ######Original Output################
        predictions tensor([[-1.2410,  1.6295, -0.5532,  0.8341, -0.3141],
        [-1.1934,  1.5816, -0.6047,  0.8289, -0.2728],
        [-1.2132,  1.5416, -0.4559,  0.8794, -0.3203],
        [-1.2115,  1.5791, -0.5111,  0.7259, -0.4447]], device='cuda:0')
        deltas_unnormalized tensor([-0.0271,  0.0395, -0.0909,  0.1492], device='cuda:0')
        state_unnormalized tensor([-0.6250,  1.2679, -2.7149,  3.7894], device='cuda:0')
        next_obs tensor([-0.6520,  1.3074, -2.8058,  3.9386], device='cuda:0')
        self.state tensor([-2.8729, -1.5044, -1.2680,  1.5962], device='cuda:0')
        uncertain False
        reward_out -0.01109282672405243

        ########SWAG Output###############
        predictions tensor([[-0.2269, -0.4017,  0.5845,  0.4366,  1.4582],
        [-0.2269, -0.4017,  0.5845,  0.4366,  1.4582],
        [-0.2269, -0.4017,  0.5845,  0.4366,  1.4582],
        [-0.2269, -0.4017,  0.5845,  0.4366,  1.4582]], device='cuda:0',
        dtype=torch.float64)
        deltas_unnormalized tensor([-0.0051, -0.0100,  0.1000,  0.0797], device='cuda:0',
            dtype=torch.float64)
        state_unnormalized tensor([ 1.1927, -0.2696, -0.0582, -1.1444], device='cuda:0')
        next_obs tensor([ 1.1877, -0.2796,  0.0418, -1.0647], device='cuda:0',
            dtype=torch.float64)
        self.state tensor([-0.8282, -3.7417,  0.0188, -0.4310], device='cuda:0',
            dtype=torch.float64)
        uncertain False
        reward_out 0.4796748161315918
        '''