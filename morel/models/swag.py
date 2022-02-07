import torch
import numpy as np
import utils
from subspaces import Subspace

class SWAG(torch.nn.Module):

    def __init__(self, base, subspace_type,
                 subspace_kwargs=None, var_clamp=1e-6, *args, **kwargs):
        super(SWAG, self).__init__()    
        self.base_model = base(*args, **kwargs)
        self.num_parameters = sum(param.numel() for param in self.base_model.parameters())

        # Initialize subspace
        if subspace_kwargs is None:
            subspace_kwargs = dict()
        self.subspace = Subspace.create(subspace_type, num_parameters=self.num_parameters,
                                        **subspace_kwargs)

        self.register_buffer('first_moment', torch.zeros(self.num_parameters))
        self.register_buffer('second_moment', torch.zeros(self.num_parameters))
        self.register_buffer('n_models', torch.zeros(1))

    
    def collect_model(self, base_model, *args, **kwargs):
        # need to refit the space after collecting a new model
        self.cov_factor = None
        new_weights = utils.flatten(base_model.parameters()) # tensor
        self.first_moment = (self.n_models.item()*self.first_moment+new_weights)/(self.n_models.item()+1) # tensor
        self.second_moment = (self.n_models.item()*self.second_moment+new_weights**2)/(self.n_models.item()+1) # tensor
        new_D = (new_weights-self.first_moment)
        
        self.subspace.collect_vector(new_D, *args, **kwargs)
        self.n_models.add_(1)

    def sample(self, scale=0.5, diag_noise=True):
        if self.cov_factor is not None:
            return
        self.cov_factor = self.subspace.get_space()
        self.var = torch.clamp(self.second_moment-self.first_moment**2, 1e-30)
        self.sqrt_var = self.var.sqrt()
                # Draw diagonal variance sample
        z1 = torch.randn_like(self.var, requires_grad=False)
        var_sample = self.sqrt_var * z1 # tensor
        z2 = torch.randn(self.cov_factor.size()[0])
        cov_sample = self.cov_factor.matmul(z2) # tensor # size (p,1)
        cov_sample = torch.flatten(cov_sample, 0) # size (p)
        rand_sample = var_sample+cov_sample #tensor # size (p)

        offset = 0
        for param in self.base_model.parameters():
            param.data.copy_(rand_sample[offset:offset + param.numel()].view(param.size()).to('cuda'))
        offset += param.numel()

        return rand_sample
        
