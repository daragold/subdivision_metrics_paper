# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:10:41 2020

@author: darac
"""
import numpy as np
import geopandas as gpd
import pandas as pd
from gerrychain import (
    Election,
    Graph,
    MarkovChain,
    Partition,
    accept,
    constraints,
    updaters,
)
from gerrychain.updaters import cut_edges
from gerrychain import GeographicPartition
import networkx as nx
import itertools

#input parameters
sample_plan_path = 'least_change_sample_plans.csv' 
num_districts = 8
state = 'MN' 
base_map = 'CONGDIST' 
 
tot_pop = 'TOTPOP' 
area = 'area'
geo_id = 'VTDID'
county_split_id = "COUNTYFIPS"
enacted = 'CONGDIST' #current US Cong CONGDIST
incumbent = '18incum'
plot_path = 'mn/MN_shapefile.shp' 
    
#read files
#initialize state_gdf
state_gdf = gpd.read_file(plot_path)
state_gdf[county_split_id] = pd.to_numeric(state_gdf[county_split_id])
sample_plans = pd.read_csv(sample_plan_path)     
state_gdf = pd.merge(state_gdf, sample_plans, on = geo_id)
graph = Graph.from_geodataframe(state_gdf)
graph.add_data(state_gdf)

total_population = state_gdf[tot_pop].sum()
ideal_population = total_population/num_districts

################### updater functions ######################
def max_pop_dev(partition, ideal_pop = ideal_population):
    #returns max deviation from ideal across all districts in partition
    dev_from_ideal = {k : abs(partition["population"][k]- ideal_pop)/ideal_pop for k in partition.parts.keys()}
    
    return max(dev_from_ideal.values())

#actual change metrics   #################################################### 
def perc_incum_precinct_match_change(partition):
     not_matched = [partition.assignment[k] != partition.assignment[partition.graph.nodes[k]["incum_node"]] \
                for k in partition.graph.nodes]   
     return sum(not_matched)/len(partition.assignment) 
 
def perc_incum_people_match_change(partition, total_pop = total_population):    
    not_matched = sum([partition.graph.nodes[k][tot_pop] for k in partition.graph.nodes if partition.assignment[k] != partition.assignment[partition.graph.nodes[k]["incum_node"]]])
    return not_matched/total_population 

def perc_precinct_change(partition):
    changed_precincts = [k for k in partition.graph.nodes if orig_assign_dict[k] != partition.assignment[k] ]
    return len(changed_precincts)/len(orig_assign_dict)

def perc_people_change(partition, total_pop = total_population):
    changed_people = sum([partition.graph.nodes[k][tot_pop] for k in partition.graph.nodes if orig_assign_dict[k] != partition.assignment[k] ])
    return changed_people/total_population
    
def perc_area_change(partition):
    changed_area = sum([partition.graph.nodes[k][area] for k in orig_assign_dict.keys() if orig_assign_dict[k] != partition.assignment[k]])
    return changed_area/sum([partition.graph.nodes[k][area] for k in partition.graph.nodes])

def perc_perim_change(partition):
    orig_perim = base_partition["perimeter"]
    new_perim = partition["perimeter"]
    perim_diff = {k: abs(orig_perim[k] - new_perim[k]) for k in new_perim.keys()}
    return sum(perim_diff.values())/sum(orig_perim.values())

def perc_county_change(partition):
    df = state_gdf.copy()
    df["current"] = df.index.map(partition.assignment.to_dict())
    old_split_counties = orig_split_counties
    new_split_counties = [i for i in np.unique(df[county_split_id]) if df.groupby(county_split_id)["current"].get_group(i).nunique() >1]   
    total_counties = len(np.unique(df[county_split_id]))
    split_unsplit = [county for county in old_split_counties if county not in new_split_counties]
    unsplit_split = [county for county in new_split_counties if county not in old_split_counties]
    county_changes = len(split_unsplit) + len(unsplit_split)
    return county_changes/total_counties

def perc_cut_edges_change(partition):
    old_cut_list = orig_cut_list
    new_cut_list = [sorted(i) for i in partition["cut_edges"]]
    cut_uncut = [edge for edge in old_cut_list if edge not in new_cut_list]
    uncut_cut = [edge for edge in new_cut_list if edge not in old_cut_list]
    edge_changes = len(uncut_cut) + len(cut_uncut)

    return edge_changes/len(graph.edges)

def perc_precinct_pair_change(partition):
    all_precinct_pairs = list(itertools.combinations(partition.graph.nodes, 2))
    change_count = 0
    for combo in all_precinct_pairs:  
        if ((orig_assign_dict[combo[0]] == orig_assign_dict[combo[1]]) and (partition.assignment[combo[0]] != partition.assignment[combo[1]])) \
        or ((orig_assign_dict[combo[0]] != orig_assign_dict[combo[1]]) and (partition.assignment[combo[0]] == partition.assignment[combo[1]])):
            change_count += 1
    return change_count/len(all_precinct_pairs)

def variation_of_info(partition, total_pop = total_population):
    VI_unnorm = 0
    for base_part in base_partition.parts:
        for compare_part in partition.parts:
            intersect_nodes = set(base_partition.parts[base_part]).intersection(set(partition.parts[compare_part]))
            if len(intersect_nodes) == 0 or sum([base_partition.graph.nodes[k][tot_pop] for k in intersect_nodes]) == 0:
                continue
            pop_intersect = sum([base_partition.graph.nodes[k][tot_pop] for k in intersect_nodes])
            pop_compare = partition["population"][compare_part]
            pop_base = base_partition["population"][base_part]
         
            VI_unnorm += (pop_intersect/total_pop)*(np.log(pop_intersect/pop_base) + np.log(pop_intersect/pop_compare))
            
    return round((-1/(2*np.log(num_districts)))*VI_unnorm,3)

def fk_index(partition):
    all_precinct_pairs = list(itertools.combinations(partition.graph.nodes, 2))
    num_pp = 0
    combo_pairs_orig = 0
    combo_pairs_new = 0
    for combo in all_precinct_pairs:  
        if ((orig_assign_dict[combo[0]] == orig_assign_dict[combo[1]]) and (partition.assignment[combo[0]] == partition.assignment[combo[1]])): \
            num_pp += 1
        
        if (orig_assign_dict[combo[0]] == orig_assign_dict[combo[1]]):
            combo_pairs_orig += 1
            
        if (partition.assignment[combo[0]] == partition.assignment[combo[1]]):
            combo_pairs_new += 1
    
    return (np.sqrt((num_pp/combo_pairs_orig)*(num_pp/combo_pairs_new)))
    
    
my_updaters = {
    "population": updaters.Tally(tot_pop, alias = "population"),
    "incumbents": updaters.Tally(incumbent, alias = "incumbents"),
    "cut_edges": cut_edges,
    "max_pop_dev": max_pop_dev,
    "perc_people_change": perc_people_change,
    "perc_area_change": perc_area_change,
    "perc_precinct_change": perc_precinct_change,
    "perc_cut_edges_change": perc_cut_edges_change,
    "perc_perim_change": perc_perim_change,
    "perc_county_change": perc_county_change,
    "perc_perim_change": perc_perim_change,
    "perc_incum_precinct_match_change": perc_incum_precinct_match_change,
    "perc_incum_people_match_change": perc_incum_people_match_change,
    "perc_precinct_pair_change": perc_precinct_pair_change,
    "variation_of_info": variation_of_info,
    "fk_index": fk_index
}

base_partition = GeographicPartition(graph = graph, assignment = base_map, updaters = my_updaters)
orig_assign_dict = {node: graph.nodes[node][base_map] for node in graph.nodes}
counties = np.unique(state_gdf[county_split_id])
orig_split_counties = [i for i in counties if state_gdf.groupby(county_split_id)[base_map].get_group(i).nunique() >1]
orig_cut_list = [sorted(i) for i in base_partition["cut_edges"]]

#set up base plan and comparator plan partitions
results_df = pd.DataFrame(columns = ['Metric'], data = ['Max Pop Dev', 'People Change', 'Area Change', 'Precinct Change', 'Perimeter Change', 'Boundary cut edge Change', 'Precinct pair change', 'County split Change','Incumbent-precinct pair change', 'Incumbent-people pair change', 'Variation of info', 'Fowlkes-Mallows Index' ])
for map_name in sample_plans.columns[1:]:
    compare_partition = GeographicPartition(graph = graph, assignment = map_name, updaters = my_updaters) 
    results_df[map_name] = [compare_partition["max_pop_dev"], compare_partition["perc_people_change"],compare_partition["perc_area_change"], compare_partition["perc_precinct_change"], compare_partition["perc_perim_change"], compare_partition["perc_cut_edges_change"], compare_partition["perc_precinct_pair_change"],compare_partition["perc_county_change"],compare_partition["perc_incum_precinct_match_change"], compare_partition["perc_incum_people_match_change"], compare_partition["variation_of_info"],compare_partition["fk_index"]]

results_df.to_csv('least_change_sample_plan_scores.csv', index = False)


