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
import os
import ast
from functools import partial


def max_pop_dev(partition):
    dev_high = max(partition.population.values())-sum(partition.population.values())/len(partition)
    dev_low = sum(partition.population.values())/len(partition) - min(partition.population.values())
    return max(dev_high,dev_low)/(sum(partition.population.values())/len(partition))

def cut_length(partition):
    return len(partition["cut_edges"])

def perc_incum_precinct_match_change(partition):
     not_matched = [partition.assignment[k] != partition.assignment[partition.graph.nodes[k]["incum_node"]] \
                for k in partition.graph.nodes]   
     return sum(not_matched)/len(partition.assignment) 
 
def perc_incum_people_match_change(partition, total_pop, tot_pop):    
    not_matched = sum([partition.graph.nodes[k][tot_pop] for k in partition.graph.nodes if partition.assignment[k] != partition.assignment[partition.graph.nodes[k]["incum_node"]]])
    return not_matched/total_pop

def perc_precinct_change(partition, orig_assign_dict):
    changed_precincts = [k for k in partition.graph.nodes if orig_assign_dict[k] != partition.assignment[k] ]
    return len(changed_precincts)/len(orig_assign_dict)

def perc_people_change(partition, total_pop, tot_pop, orig_assign_dict):
    changed_people = sum([partition.graph.nodes[k][tot_pop] for k in partition.graph.nodes if orig_assign_dict[k] != partition.assignment[k] ])
    return changed_people/total_pop
    
def perc_area_change(partition, area, orig_assign_dict):
    changed_area = sum([partition.graph.nodes[k][area] for k in orig_assign_dict.keys() if orig_assign_dict[k] != partition.assignment[k]])
    return changed_area/sum([partition.graph.nodes[k][area] for k in partition.graph.nodes])


def perim_common_refine_change(partition, map_name, state_gdf, graph, base_map , node_label, base_partition):
    state_gdf['comm_ref_assign'] = state_gdf[base_map].astype(str) + '_' + state_gdf[map_name].astype(str)  
    comm_ref_partition =  GeographicPartition(graph = graph, assignment = dict(zip((state_gdf.index if node_label == 'index' else state_gdf[node_label]), state_gdf['comm_ref_assign'])), updaters = {'cut_edges': cut_edges})
    comm_ref_perim = sum([comm_ref_partition.graph.edges[e]['shared_perim'] for e in comm_ref_partition['cut_edges']])
    base_perim = sum([base_partition.graph.edges[e]['shared_perim'] for e in base_partition['cut_edges']])
    return 1- (base_perim/comm_ref_perim)

def perimeter_change(partition, orig_assign_dict, length = True, reference = 'symmetric'):
    orig_cut_edges = [e for e in partition.graph.edges() if orig_assign_dict[e[0]]!=orig_assign_dict[e[1]]]
    retained_cut_edges = [e for e in orig_cut_edges if e in partition['cut_edges']]
    if length:
        if reference == 'orig':
            return 1-sum([partition.graph.edges[e]['shared_perim'] for e in retained_cut_edges])/sum([partition.graph.edges[e]['shared_perim'] for e in orig_cut_edges])
        if reference == 'new':
            return 1-sum([partition.graph.edges[e]['shared_perim'] for e in retained_cut_edges])/sum([partition.graph.edges[e]['shared_perim'] for e in partition['cut_edges']])
        if reference == 'symmetric':
            return ((1-sum([partition.graph.edges[e]['shared_perim'] for e in retained_cut_edges])/sum([partition.graph.edges[e]['shared_perim'] for e in orig_cut_edges])) + (1-sum([partition.graph.edges[e]['shared_perim'] for e in retained_cut_edges])/sum([partition.graph.edges[e]['shared_perim'] for e in partition['cut_edges']])))/2
    else:
        if reference == 'orig':
            return 1-len(retained_cut_edges)/len(orig_cut_edges)
        if reference == 'new':
            return 1-len(retained_cut_edges)/len(partition['cut_edges'])
        if reference == 'symmetric':
            return ((1-len(retained_cut_edges)/len(orig_cut_edges)) + (1-len(retained_cut_edges)/len(partition['cut_edges'])))/2


def perc_precinct_pair_change(partition, orig_assign_dict):
    all_precinct_pairs = list(itertools.combinations(partition.graph.nodes, 2))
    change_count = 0
    for combo in all_precinct_pairs:  
        if ((orig_assign_dict[combo[0]] == orig_assign_dict[combo[1]]) and (partition.assignment[combo[0]] != partition.assignment[combo[1]])) \
        or ((orig_assign_dict[combo[0]] != orig_assign_dict[combo[1]]) and (partition.assignment[combo[0]] == partition.assignment[combo[1]])):
            change_count += 1
    return change_count/len(all_precinct_pairs)

def variation_of_info(partition, total_pop, tot_pop, base_partition, num_districts):
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
            
    return (-1/(2*np.log(num_districts)))*VI_unnorm

def FM_index(partition, orig_assign_dict):
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
    
    return 1- (np.sqrt((num_pp/combo_pairs_orig)*(num_pp/combo_pairs_new)))
    
    
def MN_plan_report(outdir):
    #input parameters
    sample_plan_path = './input_data/least_change_plans_new.csv' 
    num_districts = 8
    base_map = 'cong_assig' #'CONGDIST' 
     
    tot_pop =  'TOTPOP' #'POP_ACS18' 
    area = 'area'
    geo_id = 'VTD' #'VTDID'
    county_split_id = 'CNTY_FIPS' #"COUNTYFIPS"
    incumbent =  'cong_incumb' #'18incum'
    plot_path = './input_data/mn20_shapefile/mn20_shapefile.shp' 
        
    #read files
    #initialize state_gdf
    state_gdf = gpd.read_file(plot_path)
    state_gdf[county_split_id] = pd.to_numeric(state_gdf[county_split_id])
    sample_plans = pd.read_csv(sample_plan_path, dtype = {geo_id: 'str'}) 
    state_gdf = pd.merge(state_gdf, sample_plans, on = geo_id)
    graph = Graph.from_geodataframe(state_gdf)
    graph.add_data(state_gdf)
    
    orig_assign_dict = {node: graph.nodes[node][base_map] for node in graph.nodes}
    total_population = state_gdf[tot_pop].sum()
    
    my_updaters = {
    "population": updaters.Tally(tot_pop, alias = "population"),
    "incumbents": updaters.Tally(incumbent, alias = "incumbents"),
    "cut_edges": cut_edges,
    "num_cut_edges": cut_length,
    "max_pop_dev": max_pop_dev, 
    "perc_people_change": partial(perc_people_change, total_pop = total_population, tot_pop = tot_pop, orig_assign_dict = orig_assign_dict),
    "perc_area_change": partial(perc_area_change, area = area, orig_assign_dict = orig_assign_dict),
    "perc_precinct_change": partial(perc_precinct_change, orig_assign_dict = orig_assign_dict),
    "perc_incum_precinct_match_change": perc_incum_precinct_match_change,
    "perc_incum_people_match_change": partial(perc_incum_people_match_change, total_pop = total_population, tot_pop = tot_pop),
    "perc_precinct_pair_change": partial(perc_precinct_pair_change,orig_assign_dict = orig_assign_dict),
    "perim_change_sym_length": partial(perimeter_change,  orig_assign_dict = orig_assign_dict, length = True, reference = 'symmetric'),
    "perim_change_sym_cut_edges": partial(perimeter_change,  orig_assign_dict = orig_assign_dict, length = False, reference = 'symmetric'), 
    "FM_index": partial(FM_index, orig_assign_dict = orig_assign_dict)
    }
    
    base_partition = GeographicPartition(graph = graph, assignment = base_map, updaters = my_updaters)
    my_updaters.update({"variation_of_info": partial(variation_of_info, total_pop = total_population, tot_pop = tot_pop, base_partition = base_partition, num_districts = num_districts)})
   #set up base plan and comparator plan partitions
    results_df = pd.DataFrame(columns = ['Metric'], data = ['Max Pop Dev', 'Num Cut Edges', 'People Change', 'Area Change', 'Precinct Change',   'Perimeter Change (Common Refinement)', 'Perimeter Change (Symmetric Length)', 'Perimeter Change (Symmetric Cut Edges)', 'Incumbent-precinct pair change', 'Incumbent-people pair change', 'Precinct pair change','Fowlkes-Mallows Index' , 'Variation of info'])#  'County split Change'])
    for map_name in sample_plans.columns[1:]:
        print("Processing:", map_name)
        my_updaters.update({"perim_comm_refine_change": partial(perim_common_refine_change, map_name = map_name, state_gdf = state_gdf, graph = graph, base_map = base_map , node_label = 'index', base_partition = base_partition)})
        compare_partition = GeographicPartition(graph = graph, assignment = map_name, updaters = my_updaters) 
        results_df[map_name] = [compare_partition["max_pop_dev"], compare_partition['num_cut_edges'], compare_partition["perc_people_change"],compare_partition["perc_area_change"], compare_partition["perc_precinct_change"], compare_partition["perim_comm_refine_change"], compare_partition["perim_change_sym_length"],  compare_partition["perim_change_sym_cut_edges"], compare_partition["perc_incum_precinct_match_change"], compare_partition["perc_incum_people_match_change"], compare_partition["perc_precinct_pair_change"],compare_partition["FM_index"], compare_partition["variation_of_info"]] #,compare_partition["perc_county_change"]  ]
    
    results_df.round(4).to_csv(outdir+'least_change_sample_MN_plan_scores.csv', index = False)


def grid_plan_report(outdir):
    adj_matrix = pd.read_csv('./input_data/least_change_grid_adjacency_matrix.csv', index_col = 0).fillna(0)
    grid_graph = nx.from_numpy_matrix(adj_matrix.to_numpy())
    node_mapping = {k:k+1 for k in grid_graph.nodes}
    G = nx.relabel_nodes(grid_graph, node_mapping)
    graph_data = pd.read_csv('./input_data/least_change_grid_plans.csv')
    edge_data = pd.read_csv('./input_data/least_change_grid_non1_edge_lengths.csv', converters={"Edge": ast.literal_eval})
    edge_data_dict = dict(zip(edge_data['Edge'], edge_data['perim']))
    
    base_map = 'BasePlan'
    num_districts = 4
    tot_pop = 'TOTPOP'
    area = 'Area'
  #  county_split_id = "COUNTYNAME"
    incumbent = 'Incumbent'
   
    for attribute in graph_data.columns:
        attribute_dict = {x:attribute for x,attribute in zip(list(graph_data['Node']),list(graph_data[attribute]))}
        for key in attribute_dict.keys():
            G.nodes[key][attribute] = attribute_dict[key]

    for edge in G.edges():
        G.edges[edge]['shared_perim'] = 1 if edge not in edge_data_dict.keys() else edge_data_dict[edge]
    
    orig_assign_dict = {node: G.nodes[node][base_map] for node in G.nodes}
    total_population = graph_data[tot_pop].sum()
    
    my_updaters = {
    "population": updaters.Tally(tot_pop, alias = "population"),
    "incumbents": updaters.Tally(incumbent, alias = "incumbents"),
    "cut_edges": cut_edges,
    "num_cut_edges": cut_length,
    "max_pop_dev": max_pop_dev,
    "perc_people_change": partial(perc_people_change, total_pop = total_population, tot_pop = tot_pop, orig_assign_dict = orig_assign_dict),
    "perc_area_change": partial(perc_area_change, area = area, orig_assign_dict = orig_assign_dict),
    "perc_precinct_change": partial(perc_precinct_change,orig_assign_dict = orig_assign_dict),
    "perc_incum_precinct_match_change": perc_incum_precinct_match_change,
    "perc_incum_people_match_change": partial(perc_incum_people_match_change, total_pop = total_population, tot_pop = tot_pop),
    "perc_precinct_pair_change": partial(perc_precinct_pair_change,orig_assign_dict = orig_assign_dict),
    "perim_change_sym_length": partial(perimeter_change,  orig_assign_dict = orig_assign_dict, length = True, reference = 'symmetric'),
    "perim_change_sym_cut_edges": partial(perimeter_change,  orig_assign_dict = orig_assign_dict, length = False, reference = 'symmetric'), 
    "FM_index": partial(FM_index, orig_assign_dict = orig_assign_dict)
    }

    base_partition = GeographicPartition(graph = G, assignment = base_map, updaters = my_updaters)
    my_updaters.update({"variation_of_info": partial(variation_of_info, total_pop = total_population, tot_pop = tot_pop, base_partition = base_partition, num_districts = num_districts)})
    results_df = pd.DataFrame(columns = ['Metric'], data = ['Max Pop Dev', 'Num Cut Edges', 'People Change', 'Area Change', 'Precinct Change',   'Perimeter Change (Common Refinement)', 'Perimeter Change (Symmetric Length)', 'Perimeter Change (Symmetric Cut Edges)', 'Incumbent-precinct pair change', 'Incumbent-people pair change', 'Precinct pair change','Fowlkes-Mallows Index' , 'Variation of info'])#  'County split Change'])

    for map_name in ['Plan1', 'Plan2', 'Plan3', 'Plan4']:
        print("Processing:", map_name)
        my_updaters.update({"perim_comm_refine_change": partial(perim_common_refine_change, map_name = map_name, state_gdf = graph_data, graph = G, base_map = base_map , node_label = 'Node', base_partition = base_partition)})
        compare_partition = GeographicPartition(graph = G, assignment = map_name, updaters = my_updaters) 
        results_df[map_name] = [compare_partition["max_pop_dev"],compare_partition['num_cut_edges'], compare_partition["perc_people_change"],compare_partition["perc_area_change"], compare_partition["perc_precinct_change"], compare_partition["perim_comm_refine_change"], compare_partition["perim_change_sym_length"],  compare_partition["perim_change_sym_cut_edges"], compare_partition["perc_incum_precinct_match_change"], compare_partition["perc_incum_people_match_change"], compare_partition["perc_precinct_pair_change"],compare_partition["FM_index"], compare_partition["variation_of_info"]] 
    
        
        results_df.round(4).to_csv(outdir+'least_change_sample_grid_plan_scores.csv', index = False)


outdir = './least_change_outputs/'
os.makedirs(os.path.dirname(outdir), exist_ok=True)
grid_plan_report(outdir)
MN_plan_report(outdir)
