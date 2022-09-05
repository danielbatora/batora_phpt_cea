import os 
import json 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.stats as st

from tqdm import tqdm 
import pickle
import copy
import re 
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def find_pairs(tree, pairs = None): 
    
    if pairs is None: 
        pairs = {}
    if "children" in tree: 
        if len(tree["children"]) == 2: 
            pairs.update({tree["children"][0]["name"] : tree["children"][1]["name"]})
        
        for child in tree["children"]: 
            find_pairs(child, pairs)
        
        
    return pairs 





def load_json(filepath): 
    with open(filepath) as json_data:
        return json.load(json_data)
    
    
def return_tree(tree, costs = None, utils = None, sub_p = None, p = None, depth = 0, decision = None, decision_option = None, fixed_costs = None, fixed_utils = None):
    if tree["type"] == "decision": 
        decision = tree["name"]
        fixed_costs = tree["fixed_costs"]
        fixed_utils = tree["fixed_utils"]
    if p is None: 
        p = []
    if sub_p is None: 
        sub_p = []
    if utils is None: 
        utils = []
    if costs is None: 
        costs = []
    if decision_option is None:     
        decision_option = []
    if type(tree) == dict and tree["type"] == "terminal":
        
        

        #print("----------")
        #print(f"Name of terminal node: {tree['name']}")
        #print(f"Cost of option: {tree['costs']}")
        #print(f"Probability of option: {round(np.prod(sub_p), 7)} {sub_p}")
        costs.append(tree["costs"] + fixed_costs)
        utils.append(tree["util"] * fixed_utils)
        p.append(np.prod(sub_p))
        decision_option.append(decision)

    elif type(tree) == dict and tree["type"] == "event":
        if not any([child["type"] == "terminal" for child in tree["children"]]):
            assert sum([child["p"] for child in tree["children"]]) == 1, f"{[child['p'] for child in tree['children']]} ,{tree['name']}, {[child['name'] for child in tree['children']]}, {decision}"

        for child in tree["children"]: 
            if depth < len(sub_p): 
                
                sub_p = sub_p[:depth]
            sub_p.append(tree["p"])
            return_tree(child, costs, utils, sub_p, p, depth = depth + 1, decision = decision, decision_option = decision_option, fixed_costs = fixed_costs, fixed_utils= fixed_utils)
    
    elif type(tree) == dict and tree["type"] != "event" and  all([child["type"] == "event" for child in tree["children"]]): 
        #print(sum([child["p"] for child in tree["children"]]))
        for child in tree["children"]:
            return_tree(child, costs, utils, sub_p, p, depth, decision = decision, decision_option = decision_option, fixed_costs = fixed_costs, fixed_utils = fixed_utils)
             
    else: 
        for child in tree["children"]: 
            return_tree(child, costs, utils, sub_p, p, depth, decision = decision, decision_option = decision_option, fixed_costs = fixed_costs, fixed_utils = fixed_utils)
        
    return costs, utils, p, decision_option
       


def randomdraw_beta(mu, sigma, size = None):
    """
    Draws 'size' number of random samples from a beta distrubution if mean is between 0 and 1
    Else draws 'size' number of random samples from a normal distributions
    
    Parameters: 
        mu: mean value between 0 and 1 
        sigma: standard deviation 
    Retunrs: 
        If size =None: a random sample 
        if size = int: a np.array of random samples
    
    """
    alpha = mu**2 * ((1 - mu) / sigma**2 - 1 / mu)
    
    beta = alpha * (1 / mu - 1)
    if mu > 1:
        shape = (mu**2)/(sigma**2)
        scale = (sigma**2) / mu
        rand = np.random.gamma(shape, scale, size)

    else: 
    
        rand = np.random.beta(alpha, beta, size)

    return rand


def simulate(num): 
    global monte_carlo    
    
    monte_carlo.dropna(axis = "index", how = "all", inplace = True)
    simulations = {}
    for ind in monte_carlo.index:
        if monte_carlo.loc[ind]["nature"] == "fixed" or monte_carlo.loc[ind]["nature"] == "choice":
            draws = [monte_carlo.loc[ind]["reference_case_value"] for i in range(num)]
        elif monte_carlo.loc[ind]["nature"] == "mean_sd":
            draws = []
            for i in range(num): 
                mean = monte_carlo.loc[ind]["reference_case_value"]
                std = monte_carlo.loc[ind]["sd"]
                draw = round(randomdraw_beta(mean, std), 9)
                draws.append(draw)           
        elif monte_carlo.loc[ind]["nature"] == "mean_ci": 
            
            draws = []
            for i in range(num): 
                std = (monte_carlo.loc[ind]["ci_up"] - monte_carlo.loc[ind]["ci_low"]) / 3.92
                mean = monte_carlo.loc[ind]["reference_case_value"]
                

                
                draw = round(randomdraw_beta(mean, std), 9)
                draws.append(draw)      
        elif  monte_carlo.loc[ind]["nature"] == "min_max":
            #VERY ROUGH ESTIMATION
            draws = []
            for i in range(num): 
                std = (monte_carlo.loc[ind]["max"] - monte_carlo.loc[ind]["min"]) / 3.92
                mean = monte_carlo.loc[ind]["reference_case_value"]
                draw = round(randomdraw_beta(mean, std), 9)
                draws.append(draw)   
                
        simulations.update({ind: draws})
        
    return simulations




def get_basecase_values(): 
    global monte_carlo    
    
    monte_carlo.dropna(axis = "index", how = "all", inplace = True)
    basecase_values = {}
    for ind in monte_carlo.index:
        if monte_carlo.loc[ind]["nature"] == "fixed" or monte_carlo.loc[ind]["nature"] == "choice" or monte_carlo.loc[ind]["nature"] == "time":
            basecase_values.update({ind:[monte_carlo.loc[ind]["reference_case_value"]]})
         
        elif monte_carlo.loc[ind]["nature"] == "mean_sd":
            basecase_values.update({ind:[monte_carlo.loc[ind]["reference_case_value"]]})

        elif monte_carlo.loc[ind]["nature"] == "mean_ci": 
            basecase_values.update({ind:[monte_carlo.loc[ind]["reference_case_value"]]})

        elif  monte_carlo.loc[ind]["nature"] == "min_max":
            basecase_values.update({ind:[monte_carlo.loc[ind]["reference_case_value"]]})        
            
    return basecase_values

def param_all_nodes(sims): 
    global node_names

    
    all_nodes = {}
    for node_name in sims:
        if "util" in node_name:
            util_node_names = [i if "util" in i else "0" for i in node_names ]
            inds  = [node_name.split("_terminal")[0]  in i for i in util_node_names]
            nodes = np.array(node_names)[inds]
            value = sims[node_name]
            for i in nodes: 
                all_nodes.update({i: value})
                
        else: 
            if node_name in node_names: 
                all_nodes.update({node_name: sims[node_name]})

    return all_nodes


def change_params_tree(tree, bc_values = False):
    global simulations, draw_num
    
    if bc_values: 
        simulations = bc_values
    
    if type(tree) == dict and tree["type"] == "event": 
        if tree["name"] in simulations: 
            tree["p"] = simulations[tree["name"]][draw_num]
        for child in tree["children"]:
            change_params_tree(child)
    else: 
        if "children" in tree: 
            for child in tree["children"]: 
                change_params_tree(child)
        
    return tree


def change_params_insitu_pairs(tree, simulations, draw_num, decision_name = None): 
    
    modality_costs = {
        "sestamibi-spect": simulations["mibi_cost"][draw_num] , 
        "fluorocholine-pet":simulations["fch_cost"][draw_num], 
        "ultrasound": simulations["us_cost"][draw_num], 
        "4dct": simulations["4dct_cost"][draw_num]}
    
    
    if tree["type"] == "decision": 
        decision_name = tree["name"].lower()
    
    if "children" in tree and len(tree["children"]) > 1:
        child_in_simulations = [child["name"] in simulations for child in tree["children"]]
        
        
        
        if all(child_in_simulations): 
            #Situation where all the children have simulated values
            inds = [i for i, x in enumerate(child_in_simulations) if x]
            for i in inds: 
                tree["children"][i]["p"] = simulations[tree["children"][i]["name"]][draw_num]
                
                

        if any(child_in_simulations): 
            #Situation in which one element has a simulated value and the other needs to be interpreted 
            ind = [i for i, x in enumerate(child_in_simulations) if x][0]
            tree["children"][ind]["p"] = simulations[tree["children"][ind]["name"]][draw_num]
            tree["children"][~ind]["p"] = 1- simulations[tree["children"][ind]["name"]][draw_num]
            

        for child in tree["children"]: 
            change_params_insitu_pairs(child, simulations, draw_num, decision_name)
    elif "children" in tree and len(tree["children"]) == 1: 

        if "no ipth" in decision_name and tree["children"][0]["name"].split("_")[0] != "second": 
            surgery_type = tree["children"][0]["name"].split("_")[0]
            tree["children"][0]["costs"] =(simulations["opcost"][draw_num] *simulations[surgery_type + "_optime"][draw_num] ) + modality_costs[decision_name.split(" ")[0]] + simulations["ipth_cost"][draw_num]
            
            
        elif "no ipth" in decision_name and tree["children"][0]["name"].split("_")[0] == "second": 

            surgery_type = tree["children"][0]["name"].split("_")[1]

            tree["children"][0]["costs"] = ((simulations["opcost"][draw_num]*simulations[surgery_type + "_optime"][draw_num] ) + modality_costs[decision_name.split(" ")[0]] + simulations["ipth_cost"][draw_num])*2 + 6445
             
            
            
        elif "with ipth" in decision_name and tree["children"][0]["name"].split("_")[0] != "second": 

            surgery_type = tree["children"][0]["name"].split("_")[0]

            tree["children"][0]["costs"] =(simulations["opcost"][draw_num] * (simulations[surgery_type + "_optime"][draw_num]+ simulations["ipth_optime"][draw_num])) + modality_costs[decision_name.split(" ")[0]]

            
        elif "with ipth" in decision_name and tree["children"][0]["name"].split("_")[0] == "second": 
            surgery_type = tree["children"][0]["name"].split("_")[1]

            tree["children"][0]["costs"] =((simulations["opcost"][draw_num] * (simulations[surgery_type + "_optime"][draw_num] + simulations["ipth_optime"][draw_num])) + modality_costs[decision_name.split(" ")[0]])*2 + 6445 

        

        if tree["children"][0]["type"] == "terminal" and  "second" in tree["children"][0]["name"] : 
            
            util_search_string = tree["children"][0]["name"].strip("second_") + "_util"
            if util_search_string in simulations: 
                tree["children"][0]["util"] = simulations[util_search_string][draw_num]
        elif tree["children"][0]["type"] == "terminal" and tree["children"][0]["name"] + "_util" in simulations:
            util_search_string = tree["children"][0]["name"] + "_util"
            tree["children"][0]["util"] = simulations[util_search_string][draw_num]

            

        
    return tree


def return_terminal_node_names(tree, term_names = None): 
    
    if term_names is None: 
        term_names = []
    
    if type(tree) == dict and tree["type"] == "terminal": 
       term_names.append(tree["name"])
    else: 
        if "children" in tree: 
            for child in tree["children"]: 
                return_terminal_node_names(child, term_names)
    return term_names




def calculate_confusion_matrix(sens, prev, spec):
    """
    Calculate TP, FN, TN, FP for a given sensitivity, prevalance and specificity

    """
    tp_fn = round(1* prev, 3)
    fp_tn = 1- tp_fn
    
    tp = tp_fn * sens
    fn = tp_fn - tp
    
    tn = fp_tn * spec
    fp = fp_tn - tn
    
    assert tp + fn + tn + fp == 1
    print(f"PPV : {tp/(tp + fp)}")
    return tp, fn, tn, fp 


def perc_of_terminal_event(event,decisions, tree, p, costs, exact_match = False):
     
    terminal_probs = list(zip(decisions, return_terminal_node_names(tree), p))

    
    event_probs = {}
    not_event_probs = {}
    for i in np.unique(decisions): 
        inds = list(np.argwhere([j == i for j in decisions]).flatten())
        
        
        
        #HOW MANY CASES OF TERMINAL EVENT 
        events = []
        not_events = []
        for j in inds : 
            if not exact_match: 
                if event in [i[1] for i in terminal_probs][j]:
                    events.append([i[2] for i in terminal_probs][j])
                else:
                    not_events.append([i[2] for i in terminal_probs][j])
            else: 
                if event == [i[1] for i in terminal_probs][j]:
                    events.append([i[2] for i in terminal_probs][j])
                else:
                    not_events.append([i[2] for i in terminal_probs][j])
        event_probs.update({i: sum(events)* 100})
        not_event_probs.update({i: sum(not_events)* 100})


    return event_probs, not_event_probs







def change_one_parameter(tree, node_name, parameter, value, add=False): 
    if "children" in tree: 
        if any([bool(re.search(node_name, child["name"])) for child in tree["children"]]) and parameter == "p":   
            ind = int(np.argwhere([bool(re.search(node_name, child["name"])) for child in tree["children"]])[0,0]) 
            tree["children"][ind][parameter] = value
            tree["children"][~ind][parameter] = 1- value
        else: 
            for child in tree["children"]: 
                    change_one_parameter(child, node_name, parameter, value, add)
    else: 
        if bool(re.search(node_name, tree["name"])): 
            if add:
                tree[parameter] = tree[parameter] + value
            else: 
                tree[parameter] = value 
            
            
    return tree




def zip_to_df(costs, utils, p, decisions): 
    df = pd.DataFrame()
    
    for i in np.unique(np.array(decisions)): 
        inds = list(np.argwhere([j == i for j in decisions]).flatten())
        total_costs = sum(np.array(costs)[inds]* np.array(p)[inds])
        total_utils = sum(np.array(utils)[inds]* np.array(p)[inds])
        
        df[i] = [total_costs, total_utils]
    
    df.index = ["Total costs", "Total utils"]
    
        
    return df 


def return_costeffective_interventions(df): 
    incremental_costs = []
    incremental_utils = []
    inmbs =  []
    inhbs = []
    for i in range(df.shape[0]): 
        if i == 0: 
            incremental_costs.append(0)
            incremental_utils.append(0)
            inmbs.append(0)
            inhbs.append(0)
        else: 
            incremental_costs.append(df["Total costs"].iloc[i] - df["Total costs"].iloc[i -1])
            incremental_utils.append(df["Total utils"].iloc[i] - df["Total utils"].iloc[i -1])
            inmbs.append(net_monetary_benefit(df["Total costs"].iloc[i], df["Total utils"].iloc[i]) - net_monetary_benefit(df["Total costs"].iloc[i-1], df["Total utils"].iloc[i-1])  )
            inhbs.append(net_health_benefit(df["Total costs"].iloc[i], df["Total utils"].iloc[i]) - net_health_benefit(df["Total costs"].iloc[i-1], df["Total utils"].iloc[i-1])  )

    df["Incremental costs"] = incremental_costs 
    df["Incremental utils"] = incremental_utils
    df["ICER"] = np.array(incremental_costs) / np.array(incremental_utils)
    df["INMB"] = inmbs
    df["INHB"] = inhbs
    
    if np.argwhere((np.array(incremental_costs) >0)  & (np.array(incremental_utils)<0)).shape[0] >0: 
        df = df.drop(df.index[np.argwhere((np.array(incremental_costs) >0)  & (np.array(incremental_utils)<0)).flatten()])
        return return_costeffective_interventions(df)
        
    else: 
        df_2 = df
        return df_2



def net_health_benefit(cost, utility): 
    return utility - cost/100000


def net_monetary_benefit(cost, utility): 
    return utility *100000 - cost


def net_monetary_benefit_wtp(cost, utility, threshold): 
    return utility * threshold - cost 



def change_param_in_csv(param, value, bc_values, tree): 
    
    copy_tree = copy.deepcopy(tree)
    bc_values[param] = [value]
    copy_tree = change_params_insitu_pairs(copy_tree,bc_values , 0)
    costs, utils, p, decisions = return_tree(copy_tree)
    df = zip_to_df(costs, utils, p, decisions)

    return df

def change_param_in_just_csv(param, value): 
    
    bc_values[param] = [value]
    
    return bc_values


def ceac(wtp_range: tuple, step=1000): 
    global modality_order, simulated_costs, simulated_utils, df_psa_ce
    


    ceacs = {}
    
    
    for ind in tqdm(range(df_psa_ce.index.shape[0] -1)): 
    
        sc = df_psa_ce.index[ind]
        alt = df_psa_ce.index[ind + 1]        
    
        sc_costs = np.array([j[np.argwhere([ i == sc for i in unique_decisions])[0,0]] for j in np.array(simulated_costs)])
        sc_utils = np.array([j[np.argwhere([ i == sc for i in unique_decisions])[0,0]] for j in np.array(simulated_utils)])
        alt_costs = np.array([j[np.argwhere([ i == alt for i in unique_decisions])[0,0]] for j in np.array(simulated_costs)])
        alt_utils = np.array([j[np.argwhere([ i == alt for i in unique_decisions])[0,0]] for j in np.array(simulated_utils)])

        icers = (alt_costs - sc_costs) / (alt_utils - sc_utils)
        
        
        probs = []
        
        for wtp in range(wtp_range[0], wtp_range[1], step): 
            prob_above = sum([i < wtp for i in icers]) / len(icers)
            probs.append(prob_above)
            
            

        ceacs.update({alt: probs})
        
        
    return ceacs