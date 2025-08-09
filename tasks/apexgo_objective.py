import torch 
import numpy as np 
import time 
import sys 
sys.path.append("../")
from tasks.utils.simple_apex_oracle.APEX_predict import apex_wrapper
from tasks.utils.peptide_vae.load_vae import load_vae, vae_forward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ApexGoObjective:
    ''' ApexGO optimization task
        Goal is to find a peptides that acheive 
        low MIC for 11 gram negative bacteria
    ''' 
    def __init__(
        self,
        dim=256,
        num_calls=0,
        dtype=torch.float32,
        lb=None, # None is important for changing vae w/ lolbo... 
        ub=None, # None is important for changing vae w/ lolbo... 
        path_to_vae_statedict="../tasks/utils/peptide_vae/checkpoints/dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118.pkl",
        max_string_length=50,
        **kwargs,
    ):
        # track total number of times the oracle has been called
        self.num_calls = num_calls
        # search space dim 
        self.dim = dim 
        # absolute upper and lower bounds on search space
        self.lb = lb
        self.ub = ub
        self.dtype = dtype
        self.path_to_vae_statedict = path_to_vae_statedict
        self.max_string_length = max_string_length
        self.predict_wrapper = apex_wrapper
        start = time.time()
        self.initialize_vae() 
        print("Time taken to initialize vae:", time.time() - start) # 19.305486917495728

    def __call__(self, xs):
        """Function defines batched function f(x) (the function we want to optimize).

        Args:
            xs (enumerable): (bsz, dim) enumerable tye of length equal to batch size (bsz), 
            each item in enumerable type must be a float tensor of shape (dim,) 
            (each is a vector in input search space).

        Returns:
            tensor: (bsz, 11) float tensor giving MIC obtained by passing each x in xs into f(x).
                    here the 11 comes from 11 gram negative bacteria 
                    
        """
        if type(xs) is np.ndarray:
            xs = torch.from_numpy(xs).to(dtype=self.dtype)
        xs = xs.to(device) # torch.Size([bsz, 256])
        aa_seqs_list = self.vae_decode(z=xs) # len bsz 
        ys = self.predict_wrapper(aa_seqs_list)
        ys = torch.from_numpy(ys).to(dtype=self.dtype) # (bsz,11)
        ys = ys*-1 # want to minimize MIC, turn into maximization problem 
        self.num_calls += len(aa_seqs_list)
        return_dict = {
            "ys":ys,
            "strings":aa_seqs_list,
        }
        return return_dict


    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        self.vae, self.dataobj = load_vae(self.path_to_vae_statedict, dim=self.dim, max_string_length=self.max_string_length)
        self.vae.to(device)
        self.vae.eval()


    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).to(dtype=self.dtype)
        self.vae.eval()
        self.vae.to(device)
        # sample strings form VAE decoder
        with torch.no_grad():
            sample = self.vae.sample(z=z.to(device).reshape(-1, 2, 128))
        # grab decoded strings
        decoded_aa_seqs = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]

        return decoded_aa_seqs

    def vae_forward_pass(self, xs_batch):
        ''' Input: 
                a list xs 
            Output: 
                z: tensor of resultant latent space codes 
                    obtained by passing the xs through the encoder
                vae_loss: the total loss of a full forward pass
                    of the batch of xs through the vae 
                    (ie reconstruction error)
        '''
        return vae_forward(xs_batch, self.dataobj, self.vae)


if __name__ == "__main__":
    obj = ApexGoObjective()
    x = torch.randn(12, 256).to(dtype=obj.dtype)
    out_dict = obj(x)
    y = out_dict["ys"]
    print(f"y: {y}") # torch.Size([12, 11]) (11 outputs for 11 different bacteria)


