import numpy as np
from language_models import HuggingFace


def insert_adv_string(msg, adv):
    return msg + adv  


def schedule_n_to_change_fixed(max_n_to_change, it):
    """ Piece-wise constant schedule for `n_to_change` (both characters and tokens) """
    # it = int(it / n_iters * 10000)

    if 0 < it <= 10:
        n_to_change = max_n_to_change
    elif 10 < it <= 25:
        n_to_change = max_n_to_change // 2
    elif 25 < it <= 50:
        n_to_change = max_n_to_change // 4
    elif 50 < it <= 100:
        n_to_change = max_n_to_change // 8
    elif 100 < it <= 500:
        n_to_change = max_n_to_change // 16
    else:
        n_to_change = max_n_to_change // 32
    
    n_to_change = max(n_to_change, 1)

    return n_to_change


def schedule_n_to_change_prob(max_n_to_change, prob, target_model):
    """ Piece-wise constant schedule for `n_to_change` based on the best prob """
    # it = int(it / n_iters * 10000)

    # need to adjust since llama and r2d2 are harder to attack
    if isinstance(target_model.model, HuggingFace):
        if 0 <= prob <= 0.01:
            n_to_change = max_n_to_change
        elif 0.01 < prob <= 0.1:
            n_to_change = max_n_to_change // 2
        elif 0.1 < prob <= 1.0:
            n_to_change = max_n_to_change // 4
        else:
            raise ValueError(f'Wrong prob {prob}')
    else:
        if 0 <= prob <= 0.1:
            n_to_change = max_n_to_change
        elif 0.1 < prob <= 0.5:
            n_to_change = max_n_to_change // 2
        elif 0.5 < prob <= 1.0:
            n_to_change = max_n_to_change // 4
        else:
            raise ValueError(f'Wrong prob {prob}')
    
    n_to_change = max(n_to_change, 1)

    return n_to_change


def extract_logprob(logprob_dict, target_token):
    logprobs = []
    if ' ' + target_token in logprob_dict:
        logprobs.append(logprob_dict[' ' + target_token])
    if target_token in logprob_dict:
        logprobs.append(logprob_dict[target_token])
    
    if logprobs == []:
        return -np.inf
    else:
        return max(logprobs)


def early_stopping_condition(best_logprobs, target_model, logprob_dict, target_token, determinstic_jailbreak, no_improvement_history=750, 
                             prob_start=0.02, no_improvement_threshold_prob=0.01):
    if determinstic_jailbreak and logprob_dict != {}:
        argmax_token = max(logprob_dict, key=logprob_dict.get)
        if argmax_token in [target_token, ' '+target_token]:
            return True
        else:
            return False
        
    if len(best_logprobs) == 0:
        return False
    
    best_logprob = best_logprobs[-1]
    if no_improvement_history < len(best_logprobs):
        prob_best, prob_history = np.exp(best_logprobs[-1]), np.exp(best_logprobs[-no_improvement_history])
        no_sufficient_improvement_condition = prob_best - prob_history < no_improvement_threshold_prob
    else: 
        no_sufficient_improvement_condition = False
    if isinstance(target_model.model, HuggingFace) and np.exp(best_logprob) > prob_start and no_sufficient_improvement_condition:
        return True 
    if isinstance(target_model.model, HuggingFace) and np.exp(best_logprob) > 0.1:
        return True  
    # for all other models
    # note: for GPT models, `best_logprob` is the maximum logprob over the randomness of the model (so it's not the "real" best logprob)
    if np.exp(best_logprob) > 0.4:  
        return True
    return False

