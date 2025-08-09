import sys 
sys.path.append("../")
sys.path.append("../../")
from constants import APEX_TEMPLATE_SEQUENCES
import torch 
from Levenshtein import distance as compute_edit_distance


def single_seq_get_perc_similarity_to_closest_tempalte(seq):
    assert type(seq) == str 
    max_perc_similarity = 0
    for template_seq in APEX_TEMPLATE_SEQUENCES:
        template_length = len(template_seq)
        edit_dist = compute_edit_distance(seq, template_seq)
        perc_similarity = (template_length - edit_dist) / template_length
        if perc_similarity > max_perc_similarity:
            max_perc_similarity = perc_similarity
    return max_perc_similarity


def get_perc_similarity_to_closest_tempalte(peptide_seqs_list, dtype=torch.float32):
    similarities = []
    for seq in peptide_seqs_list:
        max_perc_similarity = single_seq_get_perc_similarity_to_closest_tempalte(seq)
        similarities.append(max_perc_similarity)

    return torch.tensor(similarities).to(dtype=dtype)


if __name__ == "__main__":
    # test 
    peptide_seqs_list = [
        "IILFKIRLRLILRL",
        "KLKKLKLKLLKKLKLKRRRLL",
        "KKKLKLLKLKLKLLKLKL",
        "IILFLKLRLRLRILL",
    ]
    similarities = get_perc_similarity_to_closest_tempalte(peptide_seqs_list)
    print("Perc similarities to closest template:", similarities)
    print("Meets 75 constraint?", similarities>=75)
    # Perc similarities to closest template: tensor([0.2692, 0.2800, 0.3889, 0.2800])
    # Meets 75 constraint? tensor([False, False, False, False])
