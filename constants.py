SAVE_DATA_DIRECTORY = "../save_opt_data"

APEX_TEMPLATE_SEQUENCES = [
    "RACLHARSIARLHKRWRPVHQGLGLK",
    "KTLKIIRLLF",
    "KRKRGLKLATALSLNNKF",
    "KIYKKLSTPPFTLNIRTLPKVKFPK",
    "RMARNLVRYVQGLKKKKVI",
    "RNLVRYVQGLKKKKVIVIPVGIGPHANIK",
    "CVLLFSQLPAVKARGTKHRIKWNRK",
    "GHLLIHLIGKATLAL",
    "RQKNHGIHFRVLAKALR",
    "HWITINTIKLSISLKI",
]


PATH_TO_APEX_INITIALIZATION_SEQS = "../tasks/apex_init_data/init_seqs.csv"
PATH_TO_APEX_INITIALIZATION_YS = "../tasks/apex_init_data/init_ys.npy"
PATH_TO_APEX_INITIALIZATION_ZS = "../tasks/apex_init_data/init_zs.pt"
PATH_TO_APEX_INITIALIZATION_CVALS = "../tasks/apex_init_data/init_cs.csv"

def get_apex_init_best_covering_set_path(k=4):  
    return f"../tasks/apex_init_data/init_cov_set_k{k}.csv"


PATH_TO_RANO_INITIALIZATION_SEQS = "../tasks/utils/selfies_vae/guacamol_init_data.csv" # (10000, 10)
PATH_TO_RANO_INITIALIZATION_YS = "../tasks/utils/selfies_vae/rano_ys.npy" # (10000, 6)
PATH_TO_RANO_INITIALIZATION_ZS = "../tasks/utils/selfies_vae/guacamol_init_data_zs.pt" # torch.Size([20000, 256])
PATH_TO_IMG_INITIALIZATION_XS = "../tasks/utils/img_init_data/img_init_xs.npy"
PATH_TO_IMG_INITIALIZATION_YS = "../tasks/utils/img_init_data/img_init_ys.npy"
