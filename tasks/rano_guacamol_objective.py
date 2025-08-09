import torch 
import numpy as np 
import time 
import sys 
sys.path.append("../")
import selfies as sf 
from tasks.utils.selfies_vae.data import SELFIESDataset, collate_fn
from tasks.utils.selfies_vae.model_positional_unbounded import InfoTransformerVAE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# See guacamol standard benchmark tasks code here: 
# https://github.com/BenevolentAI/guacamol/blob/master/guacamol/standard_benchmarks.py
from guacamol.common_scoring_functions import TanimotoScoringFunction, RdkitScoringFunction
from guacamol.score_modifier import MaxGaussianModifier, ClippedScoreModifier, GaussianModifier
from guacamol.utils.descriptors import logP, tpsa, AtomCounter


def geometric_mean(values_list) -> float:
    """
    Computes the geometric mean of a list of values.
    """
    values = np.array(values_list)
    return values.prod() ** (1.0 / len(values))


class RanoGuacamolObjective:
    ''' Ranolazine Guacamol optimization task
        Goal is to find smiles strings that are similar 
        to Ranolazine, but also add T different extra elements 
    ''' 
    def __init__(
        self,
        dim=256,
        num_calls=0,
        dtype=torch.float32,
        lb=None, # None is important for changing vae w/ lolbo... 
        ub=None, # None is important for changing vae w/ lolbo... 
        path_to_vae_statedict="../tasks/utils/selfies_vae/selfies-vae-state-dict.pt",
        max_string_length=128,
        **kwargs,
    ):
        ranolazine = 'COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2'
        modifier = ClippedScoreModifier(upper_x=0.7)
        self.similar_to_ranolazine = TanimotoScoringFunction(ranolazine, fp_type='AP', score_modifier=modifier)
        self.logP_under_4 = RdkitScoringFunction(descriptor=logP, score_modifier=MaxGaussianModifier(mu=7, sigma=1))
        self.tpsa_f = RdkitScoringFunction(descriptor=tpsa, score_modifier=MaxGaussianModifier(mu=95, sigma=20))
        # One objective for each atom we want to try adding to Ranolazine 
        # Task 1 (adding Flourine) is the original Guacamol Ranolazine MPO task 
        # T=6 "Reactive Nonmetals" form periodic table 
        fluorine = RdkitScoringFunction(descriptor=AtomCounter('F'), score_modifier=GaussianModifier(mu=1, sigma=1.0))
        chlorine = RdkitScoringFunction(descriptor=AtomCounter('Cl'), score_modifier=GaussianModifier(mu=1, sigma=1.0))
        bromine = RdkitScoringFunction(descriptor=AtomCounter('Br'), score_modifier=GaussianModifier(mu=1, sigma=1.0))
        selenium = RdkitScoringFunction(descriptor=AtomCounter('Se'), score_modifier=GaussianModifier(mu=1, sigma=1.0))
        sulfur = RdkitScoringFunction(descriptor=AtomCounter('S'), score_modifier=GaussianModifier(mu=1, sigma=1.0))
        phosphorus = RdkitScoringFunction(descriptor=AtomCounter('P'), score_modifier=GaussianModifier(mu=1, sigma=1.0))
        self.objectives_atoms_to_add = [
            fluorine,
            chlorine,
            bromine,
            selenium,
            sulfur,
            phosphorus,
        ]

        # number of objectives 
        self.T_ = len(self.objectives_atoms_to_add) # 6
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
        start = time.time()
        self.initialize_vae() 
        print("Time taken to initialize vae:", time.time() - start) 

    def __call__(self, xs):
        """Function defines batched function f(x) (the function we want to optimize).

        Args:
            xs (enumerable): (bsz, dim) enumerable tye of length equal to batch size (bsz), 
            each item in enumerable type must be a float tensor of shape (dim,) 
            (each is a vector in input search space).

        Returns:
            tensor: (bsz, T) float tensor giving objective values obtained by 
                    passing each x in xs into each f_i(x).         
        """
        smiles_list = self.helper_xs_to_smiles(xs)
        ys = self.helper_smiles_to_scores(smiles_list=smiles_list)
        self.num_calls += len(smiles_list)
        return_dict = {
            "ys":ys,
            "strings":smiles_list,
        }
        return return_dict

    def helper_xs_to_smiles(self, xs):
        # here xs are latent zs 
        if type(xs) is np.ndarray:
            xs = torch.from_numpy(xs).to(dtype=self.dtype)
        xs = xs.to(device) # torch.Size([bsz, 256])
        seqs_list = self.vae_decode(z=xs) # len bsz list of smiles 
        return seqs_list 

    def helper_smiles_to_scores(self, smiles_list):
        N_ = len(smiles_list)
        ys = np.zeros((N_,self.T_))
        for n_ix, smile in enumerate(smiles_list):
            sim_to_rano = self.similar_to_ranolazine.score(smile)
            logp_under4 = self.logP_under_4.score(smile)
            tpsaf = self.tpsa_f.score(smile)
            # change -1's to 0.0 to avoid accidental geomean of 1.0 for all -1's
            #   guacamol -1 output means molecule was invalid 
            sim_to_rano = max(0.0, sim_to_rano) 
            logp_under4  = max(0.0, logp_under4 )
            tpsaf  = max(0.0, tpsaf)
            for t_ix, objective in enumerate(self.objectives_atoms_to_add):
                add_atom_score = objective.score(smile)
                add_atom_score = max(0.0, add_atom_score)
                geo_mean_score = geometric_mean(
                    [sim_to_rano,logp_under4,tpsaf,add_atom_score]
                )
                ys[n_ix,t_ix] = geo_mean_score
        ys = torch.from_numpy(ys).to(dtype=self.dtype) # (bsz,T)
        return ys 


    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        self.dataobj = SELFIESDataset()
        self.vae = InfoTransformerVAE(dataset=self.dataobj)
        # load in state dict of trained model:
        state_dict = torch.load(self.path_to_vae_statedict) 
        self.vae.load_state_dict(state_dict, strict=True) 
        # move to correct device 
        self.vae = self.vae.to(device)
        # put in eval mode 
        self.vae = self.vae.eval()
        # set max string length that VAE can generate
        self.vae.max_string_length = self.max_string_length


    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).to(dtype=self.dtype)
        z.to(device)
        self.vae.eval()
        self.vae.to(device)
        # sample molecular string form VAE decoder
        with torch.no_grad():
            sample = self.vae.sample(z=z.reshape(-1, 2, 128))
        # grab decoded selfies strings
        decoded_selfies = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]
        # decode selfies strings to smiles strings (SMILES is needed format for oracle)
        decoded_smiles = []
        for selfie in decoded_selfies:
            smile = sf.decoder(selfie)
            decoded_smiles.append(smile)

        return decoded_smiles

    def vae_forward_pass(self, xs_batch):
        ''' Input: 
                a list xs (smiles strings)
            Output: 
                z: tensor of resultant latent space codes 
                    obtained by passing the xs through the encoder
                vae_loss: the total loss of a full forward pass
                    of the batch of xs through the vae 
                    (ie reconstruction error)
        '''
        selfies_batch = [sf.encoder(smile) for smile in xs_batch]
        tokenized_seqs = self.dataobj.tokenize_selfies(selfies_batch)
        encoded_seqs = [self.dataobj.encode(seq).unsqueeze(0) for seq in tokenized_seqs]
        X = collate_fn(encoded_seqs)
        dict = self.vae(X.to(device))
        vae_loss, z = dict['loss'], dict['z']
        z = z.reshape(-1,256)

        return z, vae_loss
