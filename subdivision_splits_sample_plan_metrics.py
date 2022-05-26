import pandas as pd
import geopandas as gpd
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from gerrychain import Graph, GeographicPartition, updaters
from functools import partial
import os

#cardinality scores
def num_county_splits(partition, unit_df, unit_col ='VTDID', division_col = "COUNTYNAME"):
    unit_df["current"] = unit_df[unit_col].map({partition.graph.nodes[i][unit_col]:partition.assignment[i] for i in partition.assignment.keys()})
    splits = sum(unit_df.groupby(division_col)["current"].nunique() > 1)
    return splits

def num_unnec_county_splits(partition, unit_df, pop_col, ideal_population, pop_tol, unit_col ='VTDID', division_col = "COUNTYNAME"):
    unit_df["current"] = unit_df[unit_col].map({partition.graph.nodes[i][unit_col]:partition.assignment[i] for i in partition.assignment.keys()})
    splits = dict(unit_df.groupby(division_col)["current"].nunique())
    pop = dict(unit_df.groupby(division_col)[pop_col].sum())
    unnec_splits = sum([splits[key]> 1 and math.ceil(pop[key]/((1+pop_tol)*ideal_population))==1 for key in pop.keys()])
    return unnec_splits

def num_county_parts_all(partition, unit_df, unit_col ='VTDID', division_col = "COUNTYNAME"):
    unit_df["current"] = unit_df[unit_col].map({partition.graph.nodes[i][unit_col]:partition.assignment[i] for i in partition.assignment.keys()})
    parts = sum(unit_df.groupby(division_col)["current"].nunique())
    return parts

def num_county_parts_split(partition, unit_df, unit_col ='VTDID', division_col = "COUNTYNAME"):
    unit_df["current"] = unit_df[unit_col].map({partition.graph.nodes[i][unit_col]:partition.assignment[i] for i in partition.assignment.keys()})
    split_counties = pd.DataFrame(unit_df.groupby(division_col)["current"].nunique())
    parts = sum(split_counties[split_counties['current']>1]['current'])
    return parts

def num_unnec_split_parts(partition, unit_df, pop_col, ideal_population, pop_tol, unit_col ='VTDID', division_col = "COUNTYNAME"):
    unit_df["current"] = unit_df[unit_col].map({partition.graph.nodes[i][unit_col]:partition.assignment[i] for i in partition.assignment.keys()})
    splits = dict(unit_df.groupby(division_col)["current"].nunique())
    pop = dict(unit_df.groupby(division_col)[pop_col].sum())
    unnec_splits = sum([splits[key]-math.ceil(pop[key]/((1+pop_tol)*ideal_population)) for key in pop.keys()])
    return unnec_splits

def num_fragments(partition, unit_df, unit_col ='VTDID', division_col = "COUNTYNAME"):
    unit_df["current"] = unit_df[unit_col].map({partition.graph.nodes[i][unit_col]:partition.assignment[i] for i in partition.assignment.keys()})
    splits = dict(unit_df.groupby([division_col,"current"])[division_col].sum())
    node_sets = {key:[] for key in splits.keys()}
    for n in partition.graph.nodes():
        node_sets[(partition.graph.nodes[n][division_col], partition.assignment[n])].append(n)
    pieces = 0
    for key in node_sets.keys():
        subgraph = partition.graph.subgraph(node_sets[key])
        cc = nx.number_connected_components(subgraph)
        pieces += cc
    return pieces


#boundary scores
def num_split_edges(partition, division_col = "COUNTYNAME"):
    return len([e for e in partition["cut_edges"] if partition.graph.nodes[e[0]][division_col] == partition.graph.nodes[e[1]][division_col]])

def len_split_edges(partition, division_col = "COUNTYNAME"):
    return sum([partition.graph.edges[e]['shared_perim'] for e in partition["cut_edges"] if partition.graph.nodes[e[0]][division_col] == partition.graph.nodes[e[1]][division_col]])


#entroy-like scores
def even_splits_score(partition, unit_df, pop_col, unit_col ='VTDID', division_col = "COUNTYNAME"):
    unit_df["current"] = unit_df[unit_col].map({partition.graph.nodes[i][unit_col]:partition.assignment[i] for i in partition.assignment.keys()})
    splits = dict(unit_df.groupby([division_col,"current"])[pop_col].sum())
    county_pops = dict(unit_df.groupby([division_col])[pop_col].sum())
    county_parts = {county:[] for county in county_pops.keys()}
    for key in splits.keys():
        county_parts[key[0]].append(splits[key]/county_pops[key[0]])
    return sum([len(county_parts[key])*(1-min(county_parts[key])) for key in county_parts.keys()])

def root_entropy(partition, unit_df, pop_col, unit_col ='VTDID', division_col = "COUNTYNAME"):
    unit_df["current"] = unit_df[unit_col].map({partition.graph.nodes[i][unit_col]:partition.assignment[i] for i in partition.assignment.keys()})
    splits = dict(unit_df.groupby([division_col,"current"])[pop_col].sum())
    county_pops = dict(unit_df.groupby([division_col])[pop_col].sum())
    county_parts = {county:[] for county in county_pops.keys()}
    for key in splits.keys():
        county_parts[key[0]].append(splits[key]/county_pops[key[0]])
    return sum([sum([math.sqrt(item) for item in county_parts[key]]) for key in county_parts.keys()])

def shannon_entropy(partition, unit_df, pop_col, unit_col ='VTDID', division_col = "COUNTYNAME"):
    unit_df["current"] = unit_df[unit_col].map({partition.graph.nodes[i][unit_col]:partition.assignment[i] for i in partition.assignment.keys()})
    splits = dict(unit_df.groupby([division_col,"current"])[pop_col].sum())
    county_pops = dict(unit_df.groupby([division_col])[pop_col].sum())
    county_parts = {county:[] for county in county_pops.keys()}
    for key in splits.keys():
        if splits[key] > 0:
            county_parts[key[0]].append(splits[key]/county_pops[key[0]])
    return sum([sum([-1*item*math.log(item,2) for item in county_parts[key]]) for key in county_parts.keys()])


def max_pop_dev(partition):
    dev_high = max(partition.population.values())-sum(partition.population.values())/len(partition)
    dev_low = sum(partition.population.values())/len(partition) - min(partition.population.values())
    return max(dev_high,dev_low)/(sum(partition.population.values())/len(partition))

def cut_length(partition):
    return len(partition["cut_edges"])


def MN_score_report(outdir):
    #input parameters
    sample_plan_path = './input_data/subdivision_splits_plans.csv' 
    num_districts = 8
    pop_col = 'TOTPOP_10'
    pop_tol = 0
    geo_id = 'VTDID'
    plot_path = './input_data/mn_shapefile/' 
    #read files
    #initialize state_gdf
    state_gdf = gpd.read_file(plot_path)
    sample_plans = pd.read_csv(sample_plan_path)     
    state_gdf[geo_id] = state_gdf[geo_id].astype('int')
    state_gdf = pd.merge(state_gdf, sample_plans, on = geo_id)
    graph = Graph.from_geodataframe(state_gdf)
    graph.add_data(state_gdf)

    total_population = state_gdf[pop_col].sum()
    ideal_population = total_population/num_districts

    partition_updaters = {
        "population": updaters.Tally(pop_col, alias = "population"),
        "max_pop_dev": max_pop_dev,
        "num_cut_edges": cut_length,
        "num_county_splits": partial(num_county_splits,unit_df=state_gdf,unit_col=geo_id),
        "num_unnec_county_splits": partial(num_unnec_county_splits,unit_df=state_gdf,pop_col=pop_col, ideal_population=ideal_population, pop_tol=pop_tol,unit_col=geo_id),
        "num_county_parts_all": partial(num_county_parts_all,unit_df=state_gdf,unit_col=geo_id),
        "num_county_parts_split": partial(num_county_parts_split,unit_df=state_gdf,unit_col=geo_id),
        "num_unnec_split_parts": partial(num_unnec_split_parts,unit_df=state_gdf,pop_col=pop_col, ideal_population=ideal_population, pop_tol=pop_tol,unit_col=geo_id),
        "num_fragments": partial(num_fragments,unit_df=state_gdf,unit_col=geo_id),
        "num_split_edges": num_split_edges,
        "len_split_edges": len_split_edges,
        "even_splits_score": partial(even_splits_score,unit_df=state_gdf, pop_col=pop_col,unit_col=geo_id),
        "root_entropy": partial(root_entropy,unit_df=state_gdf,pop_col=pop_col,unit_col=geo_id),
        "shannon_entropy": partial(shannon_entropy,unit_df=state_gdf,pop_col=pop_col,unit_col=geo_id),
    }

    metric_list = ['max_pop_dev','num_cut_edges','num_county_splits','num_unnec_county_splits','num_county_parts_all','num_county_parts_split','num_unnec_split_parts','num_fragments','num_split_edges','len_split_edges','shannon_entropy','even_splits_score','root_entropy']

    results_df = pd.DataFrame(columns = ['Metric'], data = metric_list)
    for map_name in sample_plans.columns[1:]:
        compare_partition = GeographicPartition(graph = graph, assignment = map_name, updaters = partition_updaters) 
        out_scores = []
        for metric in metric_list:
            out_scores.append(compare_partition[metric])
        results_df[map_name] = out_scores

    results_df.round(4).to_csv(outdir+'MN_subdivision_splits_sample_plan_scores.csv', index = False)

def MN_score_report_test(outdir):
    #input parameters
    sample_plan_path = './input_data/mn20_out_plans_test.csv' 
    num_districts = 8
    pop_col = 'TOTPOP'
    pop_tol = 0
    geo_id = 'CNTY_VTD'
    county_col = 'CNTY_NAME'
    plot_path = './input_data/mn20_shapefile/' 
    #read files
    #initialize state_gdf
    state_gdf = gpd.read_file(plot_path)
    sample_plans = pd.read_csv(sample_plan_path)     
    # state_gdf[geo_id] = state_gdf[geo_id].astype('int')
    state_gdf = pd.merge(state_gdf, sample_plans, on = geo_id)
    graph = Graph.from_geodataframe(state_gdf)
    graph.add_data(state_gdf)

    total_population = state_gdf[pop_col].sum()
    ideal_population = total_population/num_districts

    partition_updaters = {
        "population": updaters.Tally(pop_col, alias = "population"),
        "max_pop_dev": max_pop_dev,
        "num_cut_edges": cut_length,
        "num_county_splits": partial(num_county_splits,unit_df=state_gdf,unit_col=geo_id, division_col = county_col),
        "num_unnec_county_splits": partial(num_unnec_county_splits,unit_df=state_gdf,pop_col=pop_col, ideal_population=ideal_population, pop_tol=pop_tol,unit_col=geo_id, division_col = county_col),
        "num_county_parts_all": partial(num_county_parts_all,unit_df=state_gdf,unit_col=geo_id, division_col = county_col),
        "num_county_parts_split": partial(num_county_parts_split,unit_df=state_gdf,unit_col=geo_id, division_col = county_col),
        "num_unnec_split_parts": partial(num_unnec_split_parts,unit_df=state_gdf,pop_col=pop_col, ideal_population=ideal_population, pop_tol=pop_tol,unit_col=geo_id, division_col = county_col),
        "num_fragments": partial(num_fragments,unit_df=state_gdf,unit_col=geo_id, division_col = county_col),
        "num_split_edges": partial(num_split_edges, division_col = county_col),
        "len_split_edges": partial(len_split_edges, division_col = county_col),
        "even_splits_score": partial(even_splits_score,unit_df=state_gdf, pop_col=pop_col,unit_col=geo_id, division_col = county_col),
        "root_entropy": partial(root_entropy,unit_df=state_gdf,pop_col=pop_col,unit_col=geo_id, division_col = county_col),
        "shannon_entropy": partial(shannon_entropy,unit_df=state_gdf,pop_col=pop_col,unit_col=geo_id, division_col = county_col),
    }

    metric_list = ['max_pop_dev','num_cut_edges','num_county_splits','num_unnec_county_splits','num_county_parts_all','num_county_parts_split','num_unnec_split_parts','num_fragments','num_split_edges','len_split_edges','shannon_entropy','even_splits_score','root_entropy']

    results_df = pd.DataFrame(columns = ['Metric'], data = metric_list)
    for map_name in sample_plans.columns[1:]:
        compare_partition = GeographicPartition(graph = graph, assignment = map_name, updaters = partition_updaters) 
        out_scores = []
        for metric in metric_list:
            out_scores.append(compare_partition[metric])
        results_df[map_name] = out_scores

    results_df.round(4).to_csv(outdir+'MN_subdivision_splits_sample_plan_scores_test.csv', index = False)


def grid_score_report(outdir):
    #input parameters
    grid_plan_path = './input_data/subdivision_splits_grid_plans.csv' 
    num_districts = 4
    pop_col = 'TOTPOP'
    pop_tol = 0
    geo_id = 'VTDID'
        
    #read files
    #initialize state_gdf
    grid_df = pd.read_csv(grid_plan_path)
    graph = nx.grid_graph(dim = (24,24))
    graph = Graph(graph)
    for attribute in [geo_id,'COUNTYNAME',pop_col,'Plan1','Plan2','Plan3','Plan4','Plan5','Plan6','Plan7']:
        attribute_dict = {(x,y):attribute for x,y,attribute in zip(list(grid_df['X']),list(grid_df['Y']),list(grid_df[attribute]))}
        for key in attribute_dict.keys():
            graph.nodes[key][attribute] = attribute_dict[key]
    for edge in graph.edges():
        graph.edges[edge]['shared_perim'] = 1

    total_population = grid_df[pop_col].sum()
    ideal_population = total_population/num_districts

    partition_updaters = {
        "population": updaters.Tally(pop_col, alias = "population"),
        "max_pop_dev": max_pop_dev,
        "num_cut_edges": cut_length,
        "num_county_splits": partial(num_county_splits,unit_df=grid_df,unit_col=geo_id),
        "num_unnec_county_splits": partial(num_unnec_county_splits,unit_df=grid_df,pop_col=pop_col, ideal_population=ideal_population, pop_tol=pop_tol,unit_col=geo_id),
        "num_county_parts_all": partial(num_county_parts_all,unit_df=grid_df,unit_col=geo_id),
        "num_county_parts_split": partial(num_county_parts_split,unit_df=grid_df,unit_col=geo_id),
        "num_unnec_split_parts": partial(num_unnec_split_parts,unit_df=grid_df,pop_col=pop_col, ideal_population=ideal_population, pop_tol=pop_tol,unit_col=geo_id),
        "num_fragments": partial(num_fragments,unit_df=grid_df,unit_col=geo_id),
        "num_split_edges": num_split_edges,
        "len_split_edges": len_split_edges,
        "even_splits_score": partial(even_splits_score,unit_df=grid_df, pop_col=pop_col,unit_col=geo_id),
        "root_entropy": partial(root_entropy,unit_df=grid_df,pop_col=pop_col,unit_col=geo_id),
        "shannon_entropy": partial(shannon_entropy,unit_df=grid_df,pop_col=pop_col,unit_col=geo_id),
    }

    metric_list = ['max_pop_dev','num_cut_edges','num_county_splits','num_unnec_county_splits','num_county_parts_all','num_county_parts_split','num_unnec_split_parts','num_fragments','num_split_edges','len_split_edges','shannon_entropy','even_splits_score','root_entropy']

    results_df = pd.DataFrame(columns = ['Metric'], data = metric_list)
    for map_name in ['Plan1', 'Plan2', 'Plan3','Plan4', 'Plan5', 'Plan6', 'Plan7']:
        compare_partition = GeographicPartition(graph = graph, assignment = map_name, updaters = partition_updaters) 
        out_scores = []
        for metric in metric_list:
            out_scores.append(compare_partition[metric])
        results_df[map_name] = out_scores

    results_df.round(4).to_csv(outdir+'grid_subdivision_splits_sample_plan_scores.csv', index = False)



outdir = './splits_outputs/'
os.makedirs(os.path.dirname(outdir), exist_ok=True)





def MN_contig_test():
    #input parameters
    sample_plan_path = './input_data/mn20_out_plans_test.csv' 
    pop_col = 'TOTPOP'
    geo_id = 'CNTY_VTD'
    plot_path = './input_data/mn20_shapefile/' 
    #read files
    #initialize state_gdf
    state_gdf = gpd.read_file(plot_path)
    sample_plans = pd.read_csv(sample_plan_path)     
    # state_gdf[geo_id] = state_gdf[geo_id].astype('int')
    state_gdf = pd.merge(state_gdf, sample_plans, on = geo_id)
    graph = Graph.from_geodataframe(state_gdf)
    graph.add_data(state_gdf)

    partition_updaters = {
        "population": updaters.Tally(pop_col, alias = "population"),
        "max_pop_dev": max_pop_dev,
        "num_cut_edges": cut_length,
    }

    for map_name in sample_plans.columns[1:]:
        compare_partition = GeographicPartition(graph = graph, assignment = map_name, updaters = partition_updaters) 
        for part in compare_partition.parts:
            subg = compare_partition.graph.subgraph(compare_partition.parts[part])
            num_cc = nx.number_connected_components(subg)
            if num_cc > 1:
                print(map_name, part, num_cc,len(subg.nodes()))
                for cc in nx.connected_components(subg):
                    print(len(cc))
                    if len(cc)<10:
                        print([subg.nodes[v][geo_id] for v in cc])




# MN_score_report(outdir)
# grid_score_report(outdir)
MN_score_report_test(outdir)
MN_contig_test()