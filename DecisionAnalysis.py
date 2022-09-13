import os 
import json 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.stats as st
from scipy.optimize import minimize
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



def get_cutoff_d(sens, ppv): 
    def confusion_matrix(params):
        
        tp, fp, fn = params
        sens_calc = tp/(tp + fn)
        ppv_calc = tp/(tp + fp)
        
        return abs(sens - sens_calc) + abs(ppv - ppv_calc)
    def const(params): 
        return params[0] + params[1] + params[2] -1
    constarnt = {'type':'eq', 'fun': const}
    results = minimize(confusion_matrix, (0.7, 0.1, 0.2), constraints = constarnt)
    
    results_array = results["x"]
    print(results_array)
    return np.round(results_array[0] + results_array[2], 2)


def get_cutoff_t(sens, ppv): 
    
    def confusion_matrix(params):
        
        tp, fp, fn = params
        sens_calc = tp/(tp + fn)
        ppv_calc = tp/(tp + fp)
        
        return abs(sens - sens_calc) + abs(ppv - ppv_calc)
    def const(params): 
        return params[0] + params[1] + params[2] -1
    constarnt = {'type':'eq', 'fun': const}
    results = minimize(confusion_matrix, (0.7, 0.1, 0.2), constraints = constarnt)
    
    results_array = results["x"]
    print(results_array)
    return np.round(results_array[0] + results_array[1], 2)



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
    
    
    for k in simulations.keys(): 
        if "a_posteriori_positive" in k: 
            sens = simulations[k.split("a_posteriori_positive")[0] + "tp"]
            ppv = simulations[k]
            
            cutoffs= []
            for j in range(len(sens)):
                cutoff = get_cutoff_d(sens[j], ppv[j])
                cutoffs.append(cutoff)
            simulations.update({k: cutoffs})
    
    
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
          
    for i in basecase_values.keys(): 
        if "a_posteriori_positive" in i: 
            sens = basecase_values[i.split("a_posteriori_positive")[0] + "tp"]
            ppv = basecase_values[i]

            cutoff = get_cutoff_d(sens, ppv)

            basecase_values.update({i: [cutoff]})
    
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
            tree["children"][0]["costs"] =(simulations["opcost"][draw_num] *simulations[surgery_type + "_optime"][draw_num] ) + simulations[surgery_type+ "_hospital"][draw_num] + modality_costs[decision_name.split(" ")[0]] + simulations["ipth_cost"][draw_num]
            
            
        elif "no ipth" in decision_name and tree["children"][0]["name"].split("_")[0] == "second": 

            surgery_type = tree["children"][0]["name"].split("_")[1]

            tree["children"][0]["costs"] = ((simulations["opcost"][draw_num]*simulations[surgery_type + "_optime"][draw_num] )+ simulations[surgery_type+ "_hospital"][draw_num] + modality_costs[decision_name.split(" ")[0]] + simulations["ipth_cost"][draw_num])*2 + 6445
             
            
            
        elif "with ipth" in decision_name and tree["children"][0]["name"].split("_")[0] != "second": 

            surgery_type = tree["children"][0]["name"].split("_")[0]

            tree["children"][0]["costs"] =(simulations["opcost"][draw_num] * (simulations[surgery_type + "_optime"][draw_num]+ simulations["ipth_optime"][draw_num]))+ simulations[surgery_type+ "_hospital"][draw_num] + modality_costs[decision_name.split(" ")[0]]

            
        elif "with ipth" in decision_name and tree["children"][0]["name"].split("_")[0] == "second": 
            surgery_type = tree["children"][0]["name"].split("_")[1]

            tree["children"][0]["costs"] =((simulations["opcost"][draw_num] * (simulations[surgery_type + "_optime"][draw_num] + simulations["ipth_optime"][draw_num]))+ simulations[surgery_type+ "_hospital"][draw_num] + modality_costs[decision_name.split(" ")[0]])*2 + 6445 

        

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


































# 1. Model Validation
"""
#Construction of a simple model comparable to the Yap et al., 2022 study
folder = r"S:\DanielB\Projects\decision_analysis"
file = "tree_2.json"
monte_carlo = pd.read_csv(os.path.join(folder, "probs_new_2020_dollars.csv"), index_col = 0)
node_names = list(pd.read_csv(os.path.join(folder, "param_names.csv"), index_col = 0).index)
#Make sure that global variable refers to 2020 dollar value monte carlo
bc_values = get_basecase_values()
#Populate all unique parameters with identical values found in the probs dataset
tree = load_json(os.path.join(folder, file))


#Parameters changed for model validation with the Yap et al., 2022 study

bc_values = change_param_in_just_csv("fp_optime", 40)
bc_values = change_param_in_just_csv("bne_optime", 125)



tree = change_params_insitu_pairs(tree,bc_values , 0)

tree = change_one_parameter(tree, "single_adenoma", "p", 1)
tree = change_one_parameter(tree, "ipth_tp", "p", 1)
tree = change_one_parameter(tree, "ipth_tn", "p", 1)

tree = change_one_parameter(tree, "single_spect_tp", "p", 0.761)
tree = change_one_parameter(tree, "single_spect_a_posteriori_positive", "p", 0.932)
tree = change_one_parameter(tree, "single_spect_tp", "p", 0.761)
tree = change_one_parameter(tree, "single_4dct_a_posteriori_positive", "p", 0.935)
tree = change_one_parameter(tree, "single_us_a_posteriori_positive", "p", 0.907)

tree = change_one_parameter(tree, "single_fch_tp", "p", 0.95)
tree = change_one_parameter(tree, "single_4dct_tp", "p", 0.894)
tree = change_one_parameter(tree, "single_us_tp", "p", 0.789)
tree = change_one_parameter(tree, "bne_rln_injury", "p", 0.01)
tree = change_one_parameter(tree, "bne_hypocalcemia_only", "p", 0.011)
tree = change_one_parameter(tree, "bne_rln_injury_plus_hypocalcemia", "p", 0.011)


tree = change_one_parameter(tree, "une", "p", 0)

costs, utils, p, decisions = return_tree(tree)

df = zip_to_df(costs, utils, p, decisions)
df_yap_sorted = df.T.sort_values("Total costs").iloc[1:, :]
df_yap_sorted = df_yap_sorted[df_yap_sorted.index.str.contains("With")]
df_yap_ce = return_costeffective_interventions(df_yap_sorted)
"""


num_of_simulations = 5000


is_linux = False
linux = "/run/user/1000/gvfs/smb-share:server=130.92.121.20,share=grpgertsch"

if is_linux: 
    
    folder = linux + "/DanielB/Projects/Decision_analysis"
    folder_fig = os.path.join(folder, "figure")
    
else: 
    
    folder = r"S:/DanielB/Projects/Decision_analysis"
    folder_fig = os.path.join(folder, "figures")

if not os.path.isdir(folder_fig): 
    os.mkdir(folder_fig)


monte_carlo = pd.read_csv(os.path.join(folder, "probs_new.csv"), index_col = 0)
node_names = list(pd.read_csv(os.path.join(folder, "param_names.csv"), index_col = 0).index)
choices = monte_carlo[monte_carlo["nature"] == "choice"]
bc_values = get_basecase_values()
file = "tree_3.json"
tree = load_json(os.path.join(folder, file))

tree = change_params_insitu_pairs(tree,bc_values , 0)
#tree = change_one_parameter(tree, "single_fch_tp", "p", 0.95)
#tree = change_one_parameter(tree, "multi_fch_tp", "p", 0.95)
#tree = change_one_parameter(tree, "ipth_tn", "p", 0.945)

costs, utils, p, decisions = return_tree(tree)

df = zip_to_df(costs, utils, p, decisions)
df_yap_sorted = df.T.sort_values("Total costs")
df_yap_sorted = df_yap_sorted[df_yap_sorted.index.str.contains("4DCT")]

df_yap_ce = return_costeffective_interventions(df_yap_sorted)











#2. Reference-case 


print("--------------")
print("2. Calculating Costs and Utilities for the reference case")
bc_values = get_basecase_values()
#Populate all unique parameters with identical values found in the probs dataset
tree = load_json(os.path.join(folder, file))
tree = change_params_insitu_pairs(tree,bc_values , 0)


costs, utils, p, decisions = return_tree(tree)

df = zip_to_df(costs, utils, p, decisions)
  

df_main = pd.DataFrame()


if choices.shape[0] > 0: 
    ipth_protocols = ["_vienna", "_miami", "_halle"]
    for i in range(len([float(i) for i in choices["choices"][0].split(",")])): 
        for j in range(choices.shape[0]): 
            event = choices.index[j]
            value = [float(i) for i in choices["choices"][j].split(",")][i]
            bc_values[event][0] = value
        tree = change_params_insitu_pairs(tree,bc_values , 0)
        costs, utils, p, decisions = return_tree(tree)
        df = zip_to_df(costs, utils, p, decisions)
        df.columns = [k + ipth_protocols[i] for k in df.columns]
    
        df_main = pd.concat([df_main, df], axis = 1)
    
cols = df_main.loc["Total costs"].drop_duplicates()
cols = list(cols.index)
df_all_ipth_protocols = df_main[cols]
df_all_ipth_protocols.to_csv(os.path.join(folder, "refcases_ipth_protocols.csv"))
    
bc_values = get_basecase_values()
tree = load_json(os.path.join(folder, file))

tree = change_params_insitu_pairs(tree,bc_values , 0)

costs, utils, p, decisions = return_tree(tree)

df = zip_to_df(costs, utils, p, decisions)

df.to_csv(os.path.join(folder, "reference_case.csv"))
df_all_sorted = df_all_ipth_protocols.T.sort_values("Total costs")
df_all_sorted.index = [i.split("_vienna")[0]if "No IPTH" in i else i  for i in df_all_sorted.index ]

incremental_costs = []
incremental_utils = []
nmbs = []
nhbs = []
for i in range(df_all_sorted.shape[0]): 
    if i == 0: 
        incremental_costs.append(0)
        incremental_utils.append(0)
        nmbs.append(0)
        nhbs.append(0)
    else: 
        incremental_costs.append(df_all_sorted["Total costs"].iloc[i] - df_all_sorted["Total costs"].iloc[i -1])
        incremental_utils.append(df_all_sorted["Total utils"].iloc[i] - df_all_sorted["Total utils"].iloc[i -1])
        nmbs.append(net_monetary_benefit(df_all_sorted["Total costs"].iloc[i], df_all_sorted["Total utils"].iloc[i]) - net_monetary_benefit(df_all_sorted["Total costs"].iloc[0], df_all_sorted["Total utils"].iloc[0]))
        nhbs.append(net_health_benefit(df_all_sorted["Total costs"].iloc[i], df_all_sorted["Total utils"].iloc[i]) - net_health_benefit(df_all_sorted["Total costs"].iloc[0], df_all_sorted["Total utils"].iloc[0]))
df_all_sorted["Incremental costs"] = incremental_costs
df_all_sorted["Incremental utils"] = incremental_utils
df_all_sorted["ICER"] = np.array(incremental_costs) / np.array(incremental_utils)
df_all_sorted["NMB"] = nmbs
df_all_sorted["NHB"] = nhbs


df_ce = return_costeffective_interventions(df_all_sorted)
     

df_all_sorted.index = df_all_sorted.index.str.replace("IPTH", "ioPTH").str.replace("_", " ").str.replace("Imaging", "").str.replace("With", "with").str.replace("No", "no")


df_all_sorted.to_csv(os.path.join(folder, "reference_case_all_interventions.csv"))
df_ce.to_csv(os.path.join(folder, "reference_case_non-dominated_interventions.csv"))




# 3. Cost-Utility scatter plots for ioPTH protocols and imaging interventions

print("3. Cost-Utility scatter plots..")
for modality in ["Ultrasound", "SPECT", "PET", "4DCT", "No IPTH"]: 
    df = df_all_ipth_protocols.loc[:, [modality in i for i in df_all_ipth_protocols.columns]]
    fig, ax = plt.subplots()
    fig.set_size_inches((7, 7))
    ax.scatter(df.loc["Total utils"], df.loc["Total costs"], color = "black", s =100)
    
    for i, txt in enumerate(df.columns): 
        ax.annotate(txt,(df.loc["Total utils"][i], df.loc["Total costs"][i]) )
    
    ax.set_xlabel("Total Utils (QALY)")
    ax.set_ylabel("Total Costs ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_fig, modality + ".eps"), format = "eps")
    plt.close()
    









#4. One-way sensitivity analysis
print("4. One-way sensitivity analysis")
param_name_dict = pd.read_csv(os.path.join(folder, "param_long_name_dictionary.csv"))
bc_values = get_basecase_values()
#Populate all unique parameters with identical values found in the probs dataset
tree = load_json(os.path.join(folder, file))
tree = change_params_insitu_pairs(tree,bc_values , 0)


costs, utils, p, decisions = return_tree(tree)
#Sensitivity analysis
params_to_ignore = ["bne_no_complications_terminal_util"]
df_all = pd.DataFrame()
ref_case = zip_to_df(costs, utils, p, decisions)
for param in tqdm(bc_values): 
    if any([i in param for i in params_to_ignore]): 
        print(f"Ignoring parameter {param}")
        continue
    else: 
        min_ = bc_values[param][0] * 0.5
        if bc_values[param][0] * 1.5 >1 and bc_values[param][0] <1: 
            max_ = 1
        else: 
            max_ = bc_values[param][0] * 1.5
        step = (max_ - min_) / 100
        dfs = pd.DataFrame()
        
        for i in np.arange(min_, max_, step):
            df = change_param_in_csv(param,i, bc_values, tree)
            df_inmb = pd.DataFrame([ net_monetary_benefit(df.loc["Total costs"].values, df.loc["Total utils"].values) - net_monetary_benefit(ref_case.loc["Total costs"].values, ref_case.loc["Total utils"].values)], index = ["inmb"], columns = ref_case.columns)
            df = pd.concat([df, df_inmb])
            df["param"] = param
            df["value"] = i
            dfs = pd.concat([dfs, df])
        
        if "cost" in param_name_dict["long_name"][param_name_dict["param"] == param].item().lower() or "time" in param_name_dict["long_name"][param_name_dict["param"] == param].item().lower():
            min_text = round(min_)
            max_text = round(max_)
        else: 
            min_text = round(min_,2)
            max_text = round(max_,2)
        dfs["param"] = [param_name_dict["long_name"][param_name_dict["param"] == i].iloc[0] + f" ({min_text}, {max_text})" for i in dfs["param"]]

        
        
        df_all = pd.concat([df_all, dfs])
        bc_values = get_basecase_values()



all_values = {}
for decision in df_all.columns[0:-2]:
    sims_per_output = df_all.loc["inmb", [decision,"param", "value" ]]
    max_values = pd.DataFrame(sims_per_output.groupby("param")[decision].agg("max"))
    min_values = pd.DataFrame(sims_per_output.groupby("param")[decision].agg("min"))
    
    max_values = max_values.loc[max_values[decision] != 0]
    min_values = min_values.loc[min_values[decision] != 0]

    min_max_values = pd.concat([max_values, min_values], join="outer", axis = 1)
    min_max_values.columns = [decision + "_max", decision + "_min"]
    all_values.update({decision:min_max_values})
    order =sims_per_output.groupby("param").agg({decision: "std"}).sort_values(by=decision, ascending = False)
    total_costs = sims_per_output[~np.array([i in order[order[decision]==0].index for i in sims_per_output["param"]])]
    order.drop(order[order[decision]==0].index, inplace = True)
    sns.set(font_scale=0.8)
    sns.catplot(y = "param", x = decision ,data=sims_per_output,color ="#A1A1A1", order = list(order.index), orient = "h", height = 8, aspect = 8/8,alpha = 0.8)
    plt.axvline(0, color = "black")
    plt.xlabel("Incremental net monetary benefit")
    plt.ylabel("")
    plt.xlim((-12500, 5000))
    plt.xticks(ticks = np.arange(-12500, 5000, 2500))
    plt.title("Range of values (lower and upper limit)")
    plt.tight_layout()

    plt.savefig(os.path.join(folder_fig, decision + "_inmb.eps"), format="eps")
    plt.close()


param_name_dict = pd.read_csv(os.path.join(folder, "param_long_name_dictionary.csv"))
#Sensitivity analysis
params_to_ignore = ["bne_no_complications_terminal_util"]
df_all = pd.DataFrame()
ref_case = zip_to_df(costs, utils, p, decisions)
for param in tqdm(bc_values): 
    if any([i in param for i in params_to_ignore]): 
        print(f"Ignoring parameter {param}")
        continue
    else: 
        min_ = bc_values[param][0] * 0.5
        if bc_values[param][0] * 1.5 >1 and bc_values[param][0] <1: 
            max_ = 1
        else: 
            max_ = bc_values[param][0] * 1.5
        step = (max_ - min_) / 100
        dfs = pd.DataFrame()
        
        for i in np.arange(min_, max_, step):
            df = change_param_in_csv(param,i, bc_values, tree)
            df_inmb = pd.DataFrame([ net_monetary_benefit(df.loc["Total costs"].values, df.loc["Total utils"].values) - net_monetary_benefit(ref_case.loc["Total costs","4DCT Imaging No IPTH"], ref_case.loc["Total utils","4DCT Imaging No IPTH"])], index = ["inmb"], columns = ref_case.columns)
            df = pd.concat([df, df_inmb])
            df["param"] = param
            df["value"] = i
            dfs = pd.concat([dfs, df])
        
        if "cost" in param_name_dict["long_name"][param_name_dict["param"] == param].item().lower() or "time" in param_name_dict["long_name"][param_name_dict["param"] == param].item().lower():
            min_text = round(min_)
            max_text = round(max_)
        else: 
            min_text = round(min_,2)
            max_text = round(max_,2)
        dfs["param"] = [param_name_dict["long_name"][param_name_dict["param"] == i].iloc[0] + f" ({min_text}, {max_text})" for i in dfs["param"]]

        
        
        df_all = pd.concat([df_all, dfs])
        bc_values = get_basecase_values()

decision = "Fluorocholine-PET Imaging No IPTH"
sims_per_output = df_all.loc["inmb", [decision,"param", "value" ]]

max_values = pd.DataFrame(sims_per_output.groupby("param")[decision].agg("max"))
min_values = pd.DataFrame(sims_per_output.groupby("param")[decision].agg("min"))

max_values = max_values.loc[max_values[decision] != 0]
min_values = min_values.loc[min_values[decision] != 0]

min_max_values = pd.concat([max_values, min_values], join="outer", axis = 1)
min_max_values.columns = [decision + "_max", decision + "_min"]


order =sims_per_output.groupby("param").agg({decision: "std"}).sort_values(by=decision, ascending = False)
total_costs = sims_per_output[~np.array([i in order[order[decision]==0].index for i in sims_per_output["param"]])]
order.drop(order[order[decision]==0].index, inplace = True)
plt.rcParams["figure.figsize"] = (20,15)
sns.set(font_scale=0.8)
sns.catplot(y = "param", x = decision ,data=sims_per_output, order = list(order.index), orient = "h", height = 8, aspect = 8/8,alpha = 0.8, color = "gray")
plt.axvline(0, color = "black")
plt.xlabel("Incremental net monetary benefit")
plt.ylabel("")
plt.xlim((-12500, 5000))
plt.xticks(ticks = np.arange(-12500, 5000, 2500))
plt.title("Range of values (lower and upper limit)")
plt.tight_layout()

plt.savefig(os.path.join(folder_fig, decision + "_4DCT.eps"), format="eps")
plt.close()

"""
#How does remaining life expectancy affect cost-effectiveness

    
cost_per_qalys = []
for i in range(1, 80): 
    cperq = (df.loc["Total costs"][2] -df.loc["Total costs"][0]) /(((df.loc["Total utils"][2] -23.688) * i) - ((df.loc["Total utils"][0] / 24) * i))
    cost_per_qalys.append(cperq)



plt.plot(cost_per_qalys, color ="k")
plt.plot([1,80], [100000, 100000], color = "r", alpha = 0.7, linestyle = "dashed")
plt.yscale("log")


net_monetary_benefit(df.loc["Total costs"], df.loc["Total utils"])



print(f"Drawing {num_of_simulations} random samples")
plt.rcParams["figure.figsize"] = (7,7)
"""


# 5. Probabilitstic uncertainty analysis
print("5. Uncertainty analysis")
simulations = simulate(num_of_simulations)


simulated_costs = []
simulated_utils = []
modalities = []
tree = load_json(os.path.join(folder, file))
for draw_num in tqdm(range(num_of_simulations)):
    
    tree = change_params_insitu_pairs(tree, simulations, draw_num)
    
    costs, utils, p, decisions = return_tree(tree)

    unique_decisions = set(decisions)
    
    costs_of_decisions = []
    utils_of_decisions = []
    modality_order = []
    for i in unique_decisions: 

        inds = list(np.argwhere([j == i for j in decisions]).flatten())
        total_costs = sum(np.array(costs)[inds]* np.array(p)[inds])
        total_utils = sum(np.array(utils)[inds]* np.array(p)[inds])
        costs_of_decisions.append(total_costs)
        utils_of_decisions.append(total_utils)
        modality_order.append(i)

    simulated_costs.append(costs_of_decisions)
    simulated_utils.append(utils_of_decisions)
    modalities.append(modality_order)

pickle.dump(simulated_costs, open(os.path.join(folder, "simulated_costs.p"), "wb"))
pickle.dump(simulated_utils, open(os.path.join(folder, "simulated_utils.p"), "wb"))
cis_cost =  st.t.interval(alpha=0.95, df=len(simulated_costs)-1, loc=np.mean(simulated_costs, axis = 0), scale=st.sem(simulated_costs)) 
cis_util =  st.t.interval(alpha=0.95, df=len(simulated_utils)-1, loc=np.mean(simulated_utils, axis = 0), scale=st.sem(simulated_utils)) 

cis_cost = tuple([i.astype(int) for i in cis_cost])
cis_util = tuple([i.round(4) for i in cis_util])

df_psa = pd.DataFrame([np.mean(simulated_costs, axis = 0), np.mean(simulated_utils, axis = 0),[(cis_cost[0][i], cis_cost[1][i]) for i in range(len(cis_cost[0]))],[(cis_util[0][i], cis_util[1][i]) for i in range(len(cis_util[0]))]],index = ["Total costs", "Total utils", "95 CIs cost", "95 CIs util"], columns = unique_decisions)
df_psa = df_psa.T.sort_values("Total costs")

incremental_costs = []
incremental_utils = []
inmbs = []
inhbs = []

for i in range(df_psa.shape[0]): 
    if i == 0: 
        incremental_costs.append(0)
        incremental_utils.append(0)
        inmbs.append(0)
        inhbs.append(0)
    else: 
        incremental_costs.append(df_psa["Total costs"].iloc[i] - df_psa["Total costs"].iloc[i -1])
        incremental_utils.append(df_psa["Total utils"].iloc[i] - df_psa["Total utils"].iloc[i -1])
        inmbs.append(net_monetary_benefit(df_psa["Total costs"].iloc[i], df_psa["Total utils"].iloc[i]) - net_monetary_benefit(df_psa["Total costs"].iloc[i-1], df_psa["Total utils"].iloc[i-1])  )
        inhbs.append(net_health_benefit(df_psa["Total costs"].iloc[i], df_psa["Total utils"].iloc[i]) - net_health_benefit(df_psa["Total costs"].iloc[i-1], df_psa["Total utils"].iloc[i-1])  )


df_psa["Incremental costs"] = incremental_costs
df_psa["Incremental utils"] = incremental_utils
df_psa["ICER"] = np.array(incremental_costs) / np.array(incremental_utils)
df_psa["INMB"] = inmbs
df_psa["INHB"] = inhbs


df_psa_ce = return_costeffective_interventions(df_psa)

df_psa.to_csv(os.path.join(folder, "psa_all_interventions.csv"))
df_psa_ce.to_csv(os.path.join(folder, "psa_nondominated_interventions.csv"))







plt.rcParams["figure.figsize"] = (6,6)

mods = ["4DCT", "Fluorocholine", "SPECT", "Ultrasound"]

fig, ax = plt.subplots(1, 1)
colors = ["r", "b", "g","k", "c", "m", "y", "gray"]
counter = 0

for mod in mods: 
    print(mod)
    decisions = [i for i, j  in enumerate(unique_decisions) if mod in j ]
    for decision in decisions:        
        x = np.array([i[decision] for i in simulated_utils])
        y = np.array([i[decision] for i in simulated_costs])
      
        
        ax.scatter(x, y, alpha = 0.05, marker = ".", color =colors[counter])
        ax.set_xlabel("Total QALYs")
        ax.set_ylabel("Total Costs ($)")
        counter += 1 
        
plt.tight_layout()
ax.set_xlim((23.8, 23.95))
ax.set_xticks([23.80, 23.82, 23.84, 23.86, 23.88, 23.90, 23.92, 23.94])
ax.tick_params(axis ="x", labelsize=15)
ax.tick_params(axis ="y", labelsize=15)
ax.set_xlabel("Total QALYs", fontsize = 15)
ax.set_ylabel("Total costs  [$]", fontsize = 15)

plt.savefig(os.path.join(folder_fig, "cost_utility_space.eps"), format="eps")
plt.close()







df_events_all = pd.DataFrame()


#6. Relevant Clinical Parameters 

print("6. relevant Clinical parameters")
#Miami protocol
bc_values = get_basecase_values()
#Populate all unique parameters with identical values found in the probs dataset
tree = load_json(os.path.join(folder, file))
tree = change_params_insitu_pairs(tree,bc_values , 0)
costs, utils, p, decisions = return_tree(tree)
events, events_no = perc_of_terminal_event("bne",decisions, tree, p, costs, exact_match = False)
title = "BNE per 1000 patients Miami"
df_events = pd.DataFrame(events.values(), events.keys(), columns = [title])
df_events.reset_index(inplace = True)
df_events.columns = ["Intervention", title]

df_events.Intervention = [i + "_miami" for i in df_events.Intervention]

sns.catplot(y ="Intervention", x = title, data = df_events, kind="bar", color = "gray", height = 5, aspect = 1)
plt.ylabel("Intervention")
plt.tight_layout()
plt.title(title)
plt.savefig(os.path.join(folder_fig,"6_" + title + ".eps"), format="eps")
plt.close()
df_events_all = pd.concat([df_events_all, df_events])


events, events_no = perc_of_terminal_event("second",decisions, tree, p, costs, exact_match = False)
title = "Second BNE per 1000 patients Miami"
df_events = pd.DataFrame(events.values(), events.keys(), columns = [title])
df_events.reset_index(inplace = True)
df_events.columns = ["Intervention", title]
df_events.Intervention = [i + "_miami" for i in df_events.Intervention]

sns.catplot(y ="Intervention", x = title, data = df_events, kind="bar", color = "gray", height = 5, aspect = 1)
plt.ylabel("Intervention")
plt.tight_layout()
plt.title(title)
plt.savefig(os.path.join(folder_fig,"6_" + title + ".eps"), format="eps")
plt.close()
df_events_all = df_events_all.merge(df_events, on =["Intervention"])


events, events_no = perc_of_terminal_event("hypocalcemia_only",decisions, tree, p, costs, exact_match = False)
title = "Persistent hypoparathyroidism per 1000 patients Miami"
df_events = pd.DataFrame(events.values(), events.keys(), columns = [title])
df_events.reset_index(inplace = True)
df_events.columns = ["Intervention", title]
df_events.Intervention = [i + "_miami" for i in df_events.Intervention]

sns.catplot(y ="Intervention", x = title, data = df_events, kind="bar", color = "gray", height = 5, aspect = 1)
plt.ylabel("Intervention")
plt.tight_layout()
plt.title(title)
plt.savefig(os.path.join(folder_fig,"6_" + title + ".eps"), format="eps")
plt.close()
df_events_all = df_events_all.merge(df_events, on =["Intervention"])

events, events_no = perc_of_terminal_event("rln_injury_only",decisions, tree, p, costs, exact_match = False)
title = "Persistent RLN injury per 1000 patients Miami"
df_events = pd.DataFrame(events.values(), events.keys(), columns = [title])
df_events.reset_index(inplace = True)
df_events.columns = ["Intervention", title]
df_events.Intervention = [i + "_miami" for i in df_events.Intervention]

sns.catplot(y ="Intervention", x = title, data = df_events, kind="bar", color = "gray", height = 5, aspect = 1)
plt.ylabel("Intervention")
plt.tight_layout()
plt.title(title)
plt.savefig(os.path.join(folder_fig,"6_" + title + ".eps"), format="eps")
plt.close()
df_events_all = df_events_all.merge(df_events, on =["Intervention"])

events, events_no = perc_of_terminal_event("rln_injury_plus_hypocalcemia",decisions, tree, p, costs, exact_match = False)
title = "Persistent RLN injury and hypoparathyroidism per 1000 patients Miami"
df_events = pd.DataFrame(events.values(), events.keys(), columns = [title])
df_events.reset_index(inplace = True)
df_events.columns = ["Intervention", title]
df_events.Intervention = [i + "_miami" for i in df_events.Intervention]

sns.catplot(y ="Intervention", x = title, data = df_events, kind="bar", color = "gray", height = 5, aspect = 1)
plt.ylabel("Intervention")
plt.tight_layout()
plt.title(title)
plt.savefig(os.path.join(folder_fig,"6_" + title + ".eps"), format="eps")
plt.close()
df_events_all = df_events_all.merge(df_events, on =["Intervention"])

#Vienna Protocol
tree = change_one_parameter(tree, "ipth_tp", "p", 0.8745)
tree = change_one_parameter(tree, "ipth_tn", "p", 0.945)
costs, utils, p, decisions = return_tree(tree)

events, events_no = perc_of_terminal_event("bne",decisions, tree, p, costs, exact_match = False)
title = "BNE per 1000 patients Vienna"
df_events = pd.DataFrame(events.values(), events.keys(), columns = [title])
df_events.reset_index(inplace = True)
df_events.columns = ["Intervention", title]
df_events.Intervention = [i + "_vienna" for i in df_events.Intervention]

sns.catplot(y ="Intervention", x = title, data = df_events, kind="bar", color = "gray", height = 5, aspect = 1)
plt.ylabel("Intervention")
plt.tight_layout()
plt.title(title)
plt.savefig(os.path.join(folder_fig,"6_" + title + ".eps"), format="eps")
plt.close()
df_events_all = pd.concat([df_events_all,df_events])

events, events_no = perc_of_terminal_event("second",decisions, tree, p, costs, exact_match = False)
title = "Second BNE per 1000 patients Vienna"
df_events = pd.DataFrame(events.values(), events.keys(), columns = [title])
df_events.reset_index(inplace = True)
df_events.columns = ["Intervention", title]
df_events.Intervention = [i + "_vienna" for i in df_events.Intervention]

sns.catplot(y ="Intervention", x = title, data = df_events, kind="bar", color = "gray", height = 5, aspect = 1)
plt.ylabel("Intervention")
plt.tight_layout()
plt.title(title)
plt.savefig(os.path.join(folder_fig,"6_" + title + ".eps"), format="eps")
plt.close()
df_events_all = df_events_all.merge(df_events, on =["Intervention"])

events, events_no = perc_of_terminal_event("hypocalcemia_only",decisions, tree, p, costs, exact_match = False)
title = "Persistent hypoparathyroidism per 1000 patients Vienna"
df_events = pd.DataFrame(events.values(), events.keys(), columns = [title])
df_events.reset_index(inplace = True)
df_events.columns = ["Intervention", title]
df_events.Intervention = [i + "_vienna" for i in df_events.Intervention]

sns.catplot(y ="Intervention", x = title, data = df_events, kind="bar", color = "gray", height = 5, aspect = 1)
plt.ylabel("Intervention")
plt.tight_layout()
plt.title(title)
plt.savefig(os.path.join(folder_fig,"6_" + title + ".eps"), format="eps")
plt.close()
df_events_all = df_events_all.merge(df_events, on =["Intervention"])

events, events_no = perc_of_terminal_event("rln_injury_only",decisions, tree, p, costs, exact_match = False)
title = "Persistent RLN injury per 1000 patients Vienna"
df_events = pd.DataFrame(events.values(), events.keys(), columns = [title])
df_events.reset_index(inplace = True)
df_events.columns = ["Intervention", title]
df_events.Intervention = [i + "_vienna" for i in df_events.Intervention]

sns.catplot(y ="Intervention", x = title, data = df_events, kind="bar", color = "gray", height = 5, aspect = 1)
plt.ylabel("Intervention")
plt.tight_layout()
plt.title(title)
plt.savefig(os.path.join(folder_fig,"6_" + title + ".eps"), format="eps")
plt.close()
df_events_all = df_events_all.merge(df_events, on =["Intervention"])

events, events_no = perc_of_terminal_event("rln_injury_plus_hypocalcemia",decisions, tree, p, costs, exact_match = False)
title = "Persistent RLN injury and hypoparathyroidism per 1000 patients Vienna"
df_events = pd.DataFrame(events.values(), events.keys(), columns = [title])
df_events.reset_index(inplace = True)
df_events.columns = ["Intervention", title]
df_events.Intervention = [i + "_vienna" for i in df_events.Intervention]

sns.catplot(y ="Intervention", x = title, data = df_events, kind="bar", color = "gray", height = 5, aspect = 1)
plt.ylabel("Intervention")
plt.tight_layout()
plt.title(title)
plt.savefig(os.path.join(folder_fig,"6_" + title + ".eps"), format="eps")
plt.close()
df_events_all = df_events_all.merge(df_events, on =["Intervention"])

#ioPTH Dual protocol

tree = change_one_parameter(tree, "ipth_tp", "p", 0.9765)
tree = change_one_parameter(tree, "ipth_tn", "p", 0.945)
costs, utils, p, decisions = return_tree(tree)

events, events_no = perc_of_terminal_event("bne",decisions, tree, p, costs, exact_match = False)
title = "BNE per 1000 patients Dual"
df_events = pd.DataFrame(events.values(), events.keys(), columns = [title])
df_events.reset_index(inplace = True)
df_events.columns = ["Intervention", title]
sns.catplot(y ="Intervention", x = title, data = df_events, kind="bar", color = "gray", height = 5, aspect = 1)
plt.ylabel("Intervention")
plt.tight_layout()
plt.title(title)
plt.savefig(os.path.join(folder_fig,"6_" + title + ".eps"), format="eps")
plt.close()
df_events_all = df_events_all.merge(df_events, on =["Intervention"])

events, events_no = perc_of_terminal_event("second",decisions, tree, p, costs, exact_match = False)
title = "Reoperations per 1000 patients Dual"
df_events = pd.DataFrame(events.values(), events.keys(), columns = [title])
df_events.reset_index(inplace = True)
df_events.columns = ["Intervention", title]
sns.catplot(y ="Intervention", x = title, data = df_events, kind="bar", color = "gray", height = 5, aspect = 1)
plt.ylabel("Intervention")
plt.tight_layout()
plt.title(title)
plt.savefig(os.path.join(folder_fig,"6_" + title + ".eps"), format="eps")
plt.close()
df_events_all = df_events_all.merge(df_events, on =["Intervention"])

events, events_no = perc_of_terminal_event("hypocalcemia_only",decisions, tree, p, costs, exact_match = False)
title = "Persistent hypoparathyroidism per 1000 patients Dual"
df_events = pd.DataFrame(events.values(), events.keys(), columns = [title])
df_events.reset_index(inplace = True)
df_events.columns = ["Intervention", title]
sns.catplot(y ="Intervention", x = title, data = df_events, kind="bar", color = "gray", height = 5, aspect = 1)
plt.ylabel("Intervention")
plt.tight_layout()
plt.title(title)
plt.savefig(os.path.join(folder_fig,"6_" + title + ".eps"), format="eps")
plt.close()
df_events_all = df_events_all.merge(df_events, on =["Intervention"])

events, events_no = perc_of_terminal_event("rln_injury_only",decisions, tree, p, costs, exact_match = False)
title = "Persistent RLN injury per 1000 patients Dual"
df_events = pd.DataFrame(events.values(), events.keys(), columns = [title])
df_events.reset_index(inplace = True)
df_events.columns = ["Intervention", title]
sns.catplot(y ="Intervention", x = title, data = df_events, kind="bar", color = "gray", height = 5, aspect = 1)
plt.ylabel("Intervention")
plt.tight_layout()
plt.title(title)
plt.savefig(os.path.join(folder_fig,"6_" + title + ".eps"), format="eps")
plt.close()
df_events_all = df_events_all.merge(df_events, on =["Intervention"])

events, events_no = perc_of_terminal_event("rln_injury_plus_hypocalcemia",decisions, tree, p, costs, exact_match = False)
title = "Persistent RLN injury and hypoparathyroidism per 1000 patients Dual"
df_events = pd.DataFrame(events.values(), events.keys(), columns = [title])
df_events.reset_index(inplace = True)
df_events.columns = ["Intervention", title]
sns.catplot(y ="Intervention", x = title, data = df_events, kind="bar", color = "gray", height = 5, aspect = 1)
plt.ylabel("Intervention")
plt.tight_layout()
plt.title(title)
plt.savefig(os.path.join(folder_fig,"6_" + title + ".eps"), format="eps")
plt.close()
df_events_all = df_events_all.merge(df_events, on =["Intervention"])


df_events_all.iloc[:, 1:] = df_events_all.iloc[:, 1:] * 10

df_events_all["Intervention"] = df_events_all["Intervention"].str.replace("IPTH", "ioPTH")
df_events_all["Intervention"] = df_events_all["Intervention"].str.replace("4DCT", "4D-CT")
df_events_all["Intervention"] = df_events_all["Intervention"].str.replace("Fluorocholine-PET", "FCH-PET/CT")



df_events_all =df_events_all.round(3)


df_events_ipth =  df_events_all.iloc[:, 1:].loc[[i %2==1 for i in  range(df_events_all.shape[0])]].reset_index()

df_events_no_ipth =  df_events_all.iloc[:, 1:].loc[[i %2==0 for i in  range(df_events_all.shape[0])]].reset_index()


df_events_delta = df_events_ipth.subtract(df_events_no_ipth).drop("index", axis = 1)

df_events_delta.index = ["4D-CT", "FCH-PET/CT", "Sestamibi-SPECT", "Ultrasound"]
df_events_delta.to_csv(os.path.join(folder, "clinical_outcomes_difference.csv"))
df_events_all.to_csv(os.path.join(folder, "clinical_outcomes.csv"), index = 0)


df_events_delta = df_events_delta.T.sort_values("4D-CT", axis = 0, ascending=False)

df_events_delta.reset_index(inplace =True)


df_events_delta = pd.read_csv(os.path.join(folder, "clinical_outcomes_difference.csv"), index_col = 0)



style = sns.axes_style()

style["grid.color"] = "#EAEAF2"
style["axes.facecolor"] = "white"
plt.rcParams["figure.figsize"] = (6,6)

sns.set_theme(style = style)
sns.set(font_scale=1.5)

sns.catplot(y="index",x = "4D-CT", data =df_events_delta, kind="bar", color = "gray", height = 8, aspect = 12/8, edgecolor = "black")
plt.xlabel("# of patients compared to no ioPTH")
plt.ylabel("")
plt.tight_layout()

plt.savefig(os.path.join(folder_fig,"6_difference.eps"), format="eps")





"""

if "PET" in modality_order[0]: 
    pet_order = 0
    spect_order = 3
else: 
    pet_order = 1
    spect_order = 0




print("________________________________________")
print(f"Estimated cost of PET:             {int(np.mean(np.array([i[pet_order] for i in simulated_costs])))}  +-  {int(np.std(np.array([i[pet_order] for i in simulated_costs])))}  ")
print(f"Estimated utility of PET:          {round(np.mean(np.array([i[pet_order] for i in simulated_utils])), 3)}  +-  {round(np.std(np.array([i[pet_order] for i in simulated_utils])), 3)}  ")
print(f"Estimated $/QUALY for PET:       {int(np.mean(np.array([i[pet_order] for i in simulated_costs]) / np.array([i[pet_order] for i in simulated_utils])))}")




print("________________________________________")
print(f"Estimated cost of SPECT:             {int(np.mean(np.array([i[spect_order] for i in simulated_costs])))}  +-  {int(np.std(np.array([i[spect_order] for i in simulated_costs])))}  ")
print(f"Estimated utility of SPECT:          {round(np.median(np.array([i[spect_order] for i in simulated_utils])), 3)}  +-  {round(np.std(np.array([i[spect_order] for i in simulated_utils])), 3)}  ")
print(f"Estimated $/QUALY for SPECT:         {int(np.mean(np.array([i[spect_order] for i in simulated_costs]) / np.array([i[spect_order] for i in simulated_utils])))}")

#Incremental cost and utility plot

incremental_costs = np.array([cost[pet_order] - cost[spect_order] for cost in simulated_costs])
incremental_utils = np.array([util[pet_order] - util[spect_order] for util in simulated_utils])
cost_per_util = incremental_costs / incremental_utils
cost_per_util_focus = cost_per_util[(cost_per_util>0) & (cost_per_util< 100000)]
modality_better = [1 for i in range(len(cost_per_util)) if incremental_costs[i] < 0 and incremental_utils[i] > 0 ]
modality_worse = [2 for i in range(len(cost_per_util)) if incremental_costs[i] > 0 and incremental_utils[i] < 0 ]
modality_above_wtp = [3 for i in range(len(cost_per_util)) if cost_per_util[i] > 100000 ]

np.median(cost_per_util)
plt.hist(cost_per_util, range=(-100000, 1000000), bins = 100)
nbins = 300
k = kde.gaussian_kde([incremental_utils, incremental_costs])
xi, yi = np.mgrid[incremental_utils.min():incremental_utils.max():nbins*1j, incremental_costs.min():incremental_costs.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
fig, ax = plt.subplots(3, 1)
fig.set_size_inches(8, 8)


ax[0].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap = "bwr", shading="nearest")
#ax[0].plot(np.linspace(np.min(xi), np.max(xi), 10), [0 for i in range(10)], color = "black", linestyle = "dashed")
#ax[0].plot([0 for i in range(10)],np.linspace(np.min(yi), np.max(yi), 10), color = "black", linestyle = "dashed" )
ax[1].hist(cost_per_util_focus, bins = 100, color = "gray", edgecolor = "black")    
ax[2].bar([1, 2, 3, 4], [len(cost_per_util_focus), len(modality_better), len(modality_worse), len(modality_above_wtp)] , color = ["gray", "red", "blue", "green"],edgecolor = "black",  tick_label = ["ICER < WTP", "More QUALY, less $", "Less QUALY, more$", "ICER > WTP"])
fig.tight_layout()



ax[1].set_xlabel("ICER")
ax[1].set_ylabel("Count")
ax[0].set_xlabel("Incremental QUALYs")
ax[0].set_ylabel("Incremental Costs ($)")
ax[2].set_ylabel("Count")

plt.savefig(os.path.join(folder, "icer_monte_carlo.eps"),format="eps")
"""
        
        
        
        
        
        
        
#7.  Cost effectiveness Acceptability Curve 
print("7. Constructing Cost effectiveness Acceptability Curve ")
        
wtp_range = (20000, 1000000)        
ceacs = ceac(wtp_range)
        
for i in ceacs: 
    probs = ceacs[i]
    
    plt.plot(range(wtp_range[0], wtp_range[1], 1000), probs, color = "black")
    plt.title(i)
    plt.xlabel("WTP Threshold", fontsize = 20)
    plt.ylabel("p CE", fontsize = 20)
    plt.savefig(os.path.join(folder_fig, f"7_{i}_ceac.eps"), format="eps")
    plt.close()

    

#7/2. CEAC curve comparing net monetary benefit parameters 



df_ceac = pd.DataFrame()
for wtp in tqdm(range(wtp_range[0], wtp_range[1], 1000)):
    nmbs = []
    for i in range(len(simulated_costs)): 
        
        nmb = np.argmax([net_monetary_benefit_wtp(simulated_costs[i][j], simulated_utils[i][j], wtp) for j in range(len(simulated_costs[i]))])
    
        nmbs.append(nmb)
        
    unique, counts = np.unique(nmbs, return_counts = True)
    nmb_perc = []
    for k in range(len(modality_order)): 
        if k in unique:    
            perc = counts[np.argwhere(unique == k)].item() / len(nmbs)
            nmb_perc.append(perc)
        else: 
            nmb_perc.append(0)
    df_perc = pd.DataFrame([nmb_perc], columns = modality_order, index = [wtp])
    df_ceac = pd.concat([df_ceac, df_perc])
        
    
    
df_ceac = df_ceac.loc[: ,["No" in i for i in df_ceac.columns]]

plt.rcParams["figure.figsize"] = (7,7)
sns.set_theme()

df_ceac.plot(fontsize = 15, legend = None)
plt.xlabel("WTP Threshold",fontsize = 15)
plt.ylabel("Proportion of highest net monetary benefit", fontsize = 15)
plt.savefig(os.path.join(folder_fig, f"7_2_ceac_nmb.eps"), format="eps")


#8.  Threshold analysis 
param_name_dict = pd.read_csv(os.path.join(folder, "param_long_name_dictionary.csv"))

print("8. Threshold analysis")
params_to_ignore = []
bc_values = get_basecase_values()
df_all = pd.DataFrame()
for param in tqdm(bc_values): 
    if any([i in param for i in params_to_ignore]): 
        print(f"Ignoring parameter {param}")
        continue
    else: 
        min_ =  0

        if  bc_values[param][0] <1: 
            max_ = 1
        else: 
            max_ = bc_values[param][0]
        
        step = (max_ - min_) / 100
        dfs = pd.DataFrame()
        
        for i in np.arange(min_, max_, step):
            df = change_param_in_csv(param,i, bc_values, tree)
            df["param"] = param
            df["value"] = i
            
            dfs = pd.concat([dfs, df])
        
        df_all = pd.concat([df_all, dfs])
        
        bc_values = get_basecase_values()

df_all["param"] = [param_name_dict["long_name"][param_name_dict["param"] == i].iloc[0] + f" ({round(df_all[df_all['param'] == i]['value'].min(),3)}, {round(df_all[df_all['param'] == i]['value'].max(),3)})" for i in df_all["param"]]
df_inmb_compare = df_all.loc["Total costs"].iloc[:, -2:]


style = sns.axes_style()

style["grid.color"] = "#EAEAF2"
style["axes.facecolor"] = "white"
style.update({"grid.linewidth": 20 })

sns.set_theme(style = style)
for mod in ["4dct", "fluorocholine-pet", "sestamibi-spect", "ultrasound"]:
    df_inmb_compare = df_all.loc["Total costs"].iloc[:, -2:]



    num_1 = np.argwhere(df_all.columns.str.lower().str.contains(mod) & df_all.columns.str.lower().str.contains("with")).item()
    num_2 = np.argwhere(df_all.columns.str.lower().str.contains(mod) & df_all.columns.str.lower().str.contains("no ipth")).item()
    
    df_inmb_compare["inmb"] = (net_monetary_benefit(df_all.loc["Total costs"].iloc[:,[num_1]].values , df_all.loc["Total utils"].iloc[:,[num_1]].values) - net_monetary_benefit(df_all.loc["Total costs"].iloc[:,[num_2]].values , df_all.loc["Total utils"].iloc[:,[num_2]].values))

    order =df_inmb_compare.groupby("param").agg({"inmb": "max"}).sort_values(by="inmb", ascending = False)
    inmns_increase = df_inmb_compare[np.array([i in order[order["inmb"]>0].index for i in df_inmb_compare["param"]])]
   
    order.drop(order[order["inmb"]<0].index, inplace = True)
    
    
    new_names = []
    order_new_names =[]
    for param in inmns_increase.param.unique():
        slice_ = inmns_increase[(inmns_increase.param == param) & (inmns_increase.inmb > 0)]
        lowest_ind = slice_.value.iloc[np.argmin(slice_.inmb.values)]
        df_size = inmns_increase[(inmns_increase.param == param)].param.shape[0]
        
        new_names.extend([slice_.param.iloc[0].split("(")[0] + f"({lowest_ind.round(2)})" for i in range(df_size)])
        order_new_names.append(slice_.param.iloc[0].split("(")[0] + f"({lowest_ind.round(2)})")
    inmns_increase.param= new_names
    order.index = order_new_names
    
    
    plt.rcParams["figure.figsize"] = (20,3)
    
    sns.catplot(y = "param", x = "inmb" ,data=inmns_increase,alpha = 0.8, order = list(order.index), orient = "h", height = 7, aspect = 7/7, color = "gray")
    plt.axvline(0, color = "black")
    plt.xlabel("Incremental Net Monetary Benefit")
    plt.title(mod)
    plt.ylabel("Parameter and range of values (lower-upper bound")
    plt.tight_layout()
    
    plt.savefig(os.path.join(folder_fig, f"8_params_pos_influence_on_ipth_{mod}.eps"), format="eps")
    plt.close()
        
        
    for feature in inmns_increase.param.unique(): 
        plt.rcParams["figure.figsize"] = (6,6)
        plt.rcParams["font.size"] = 18
        df_one_feature = inmns_increase[inmns_increase.param == feature]
        nmb = df_one_feature["inmb"]
        
        if all(nmb<0): 
            plt.plot(df_one_feature["value"], nmb, color = "k")
        else: 
            fig, ax = plt.subplots()
            ind_lowest = np.argmin(np.array([i if i > 0 else 500000 for i in nmb], dtype = "object"))    
            ax.plot(df_one_feature["value"], nmb, color = "k")
            ax.axvline(df_one_feature["value"][ind_lowest], color ="gray",linestyle="dotted")
            
        
        ax.set_xlabel(feature.split("(")[0] + f"({df_one_feature['value'][ind_lowest].round(2)})", fontsize = 15)
        ax.set_ylabel("INMB", fontsize = 15)
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

        #plt.xticks(np.linspace(df_one_feature["value"].min(), df_one_feature["value"].max(), 6), labels = [round(i, 2) for i in np.linspace(df_one_feature["value"].min(), df_one_feature["value"].max(), 6)])
        #plt.yticks(np.linspace(nmb.min(), nmb.max(), 6), labels = [round(i) for i in np.linspace(nmb.min(), nmb.max(), 6)])
       

        plt.tight_layout()
        plt.savefig(os.path.join(folder_fig, f"8_{mod}_{feature}_detailed.eps"), format="eps")
        plt.close()
            
        
        
        
df_inmb_compare = df_all.loc["Total costs"].iloc[:, -2:]



num_1 = np.argwhere(df_all.columns.str.lower().str.contains("fluorocholine-pet") & df_all.columns.str.lower().str.contains("no ipth")).item()
num_2 = np.argwhere(df_all.columns.str.lower().str.contains("4dct") & df_all.columns.str.lower().str.contains("no ipth")).item()

df_inmb_compare["inmb"] = (net_monetary_benefit(df_all.loc["Total costs"].iloc[:,[num_1]].values , df_all.loc["Total utils"].iloc[:,[num_1]].values) - net_monetary_benefit(df_all.loc["Total costs"].iloc[:,[num_2]].values , df_all.loc["Total utils"].iloc[:,[num_2]].values))

order =df_inmb_compare.groupby("param").agg({"inmb": "max"}).sort_values(by="inmb", ascending = False)
inmns_increase = df_inmb_compare[np.array([i in order[order["inmb"]>0].index and "4D-CT" not in i  for i in df_inmb_compare["param"]])]

order.drop(order[order["inmb"]<0].index, inplace = True)
order = order[["4D-CT" not in i for i in order.index]]


new_names = []
order_new_names =[]
for param in inmns_increase.param.unique():
    slice_ = inmns_increase[(inmns_increase.param == param) & (inmns_increase.inmb > 0)]
    lowest_ind = slice_.value.iloc[np.argmin(slice_.inmb.values)]
    df_size = inmns_increase[(inmns_increase.param == param)].param.shape[0]
    
    new_names.extend([slice_.param.iloc[0].split("(")[0] + f"({lowest_ind.round(2)})" for i in range(df_size)])
    order_new_names.append(slice_.param.iloc[0].split("(")[0] + f"({lowest_ind.round(2)})")
inmns_increase.param= new_names
order.index = order_new_names




plt.rcParams["figure.figsize"] = (20,3)

sns.catplot(y = "param", x = "inmb" ,data=inmns_increase,alpha = 0.8, order = list(order.index), orient = "h", height = 7, aspect = 7/7, color = "gray")
plt.axvline(0, color = "black")
plt.xlabel("Incremental Net Monetary Benefit")
plt.title("FCH to 4DCT")
plt.ylabel("Parameter and range of values (lower-upper bound")
plt.tight_layout()

plt.savefig(os.path.join(folder_fig, f"8_params_pos_influence_on_pet_CT.eps"), format="eps")
plt.close()
    
    
for feature in inmns_increase.param.unique(): 
    plt.rcParams["figure.figsize"] = (6,6)
    plt.rcParams["font.size"] = 18
    df_one_feature = inmns_increase[inmns_increase.param == feature]
    nmb = df_one_feature["inmb"]
    
    if all(nmb<0): 
        plt.plot(df_one_feature["value"], nmb, color = "k")
    else: 
        fig, ax = plt.subplots()
        ind_lowest = np.argmin(np.array([i if i > 0 else 500000 for i in nmb], dtype = "object"))    
        ax.plot(df_one_feature["value"], nmb, color = "k")
        ax.axvline(df_one_feature["value"][ind_lowest], color ="gray",linestyle="dotted")
        
    
    ax.set_xlabel(feature.split("(")[0] + f"({df_one_feature['value'][ind_lowest].round(2)})", fontsize = 15)
    ax.set_ylabel("INMB", fontsize = 15)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    #plt.xticks(np.linspace(df_one_feature["value"].min(), df_one_feature["value"].max(), 6), labels = [round(i, 2) for i in np.linspace(df_one_feature["value"].min(), df_one_feature["value"].max(), 6)])
    #plt.yticks(np.linspace(nmb.min(), nmb.max(), 6), labels = [round(i) for i in np.linspace(nmb.min(), nmb.max(), 6)])
   

    plt.tight_layout()
    plt.savefig(os.path.join(folder_fig, f"8_{mod}_{feature}_detailed.eps"), format="eps")
    plt.close()
       
        
             
        