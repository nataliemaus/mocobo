import sys 
sys.path.append("../")
from tasks.utils.peptide_vae.vae import InfoTransformerVAE 
from tasks.utils.peptide_vae.data import DataModuleKmers, collate_fn
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# example function to load vae, loads uniref vae 
def load_vae(
    path_to_vae_statedict,
    dim=256, # dim//2
    max_string_length=50,
):
    data_module = DataModuleKmers(
        batch_size=10,
        k=1,
        load_data=False,
    )
    dataobj = data_module.train
    vae = InfoTransformerVAE(
        dataset=dataobj, 
        d_model=dim//2,
        kl_factor=0.0001,
        encoder_dim_feedforward=256,
        decoder_dim_feedforward=256,
        encoder_num_layers=6,
        decoder_num_layers=6,
    ) 

    # load in state dict of trained model:
    if path_to_vae_statedict:
        if torch.cuda.is_available():
            state_dict = torch.load(path_to_vae_statedict) 
        else:
            state_dict = torch.load(path_to_vae_statedict, map_location=torch.device('cpu')) 
        vae.load_state_dict(state_dict, strict=True) 
    vae = vae.to(device)
    vae = vae.eval()

    # set max string length that VAE can generate
    vae.max_string_length = max_string_length

    return vae, dataobj 


def vae_forward(xs_batch, dataobj, vae):
    ''' Input: 
            a list xs 
        Output: 
            z: tensor of resultant latent space codes 
                obtained by passing the xs through the encoder
            vae_loss: the total loss of a full forward pass
                of the batch of xs through the vae 
                (ie reconstruction error)
    '''
    # assumes xs_batch is a batch of smiles strings 
    tokenized_seqs = dataobj.tokenize_sequence(xs_batch)
    encoded_seqs = [dataobj.encode(seq).unsqueeze(0) for seq in tokenized_seqs]
    X = collate_fn(encoded_seqs)
    dict = vae(X.to(device))
    vae_loss, z = dict['loss'], dict['z']
    z = z.reshape(-1,256)

    return z, vae_loss


def vae_decode(z, vae, dataobj):
    '''Input
            z: a tensor latent space points (bsz, self.dim)
        Output
            a corresponding list of the decoded input space 
            items output by vae decoder 
    '''
    z = z.to(device)
    vae = vae.eval()
    vae = vae.to(device)
    # sample molecular string form VAE decoder
    sample = vae.sample(z=z.reshape(-1, 2, 128))
    # grab decoded aa strings
    decoded_seqs = [dataobj.decode(sample[i]) for i in range(sample.size(-2))]

    return decoded_seqs