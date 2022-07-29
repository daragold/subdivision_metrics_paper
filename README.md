### Subsivision-Split and Least-Change Metrics

This repo contains the code and data to supplement "The Gameability of Redistricting Criteria"


All data is found in the input_data directory: 
- **mn20_shapefile** is a shapefile of the Minnesota 2020 VTDs used for our example Minnesota plans and is used both in metric calculations and in our map script
- **wi20_shapefile** is a shapefile of the Wisconsin 2020 Wards used for our example Wisconsin plans and is used both in metric calculations and in our map script
- **subdivision_splits_grid_plans.csv** provides the sample grid examples used to demonstrate the subdivision-split metrics
- **least_change_grid_plans.csv**, **least_change_grid_non1_edge_lengths.csv**, and **least_change_grid_adjacency_matrix.csv** provide the sample grid examples used to demonstrate the least-change metrics
- **mn_sample_plans.csv** provides the plan assignments for our example Minnesota congressional plans 
- **wi_proposed_plans.csv** provides the plan assignments for our example Wisconsin congressional plans 

The code is run in three different python scripts:
- **subdivision_splits_sample_plan_metrics.py** runs the county-split metrics on the grid, Minnesota, and Wisconsin example plans and outputs scores to a **splits_outputs** directory
- **least_change_sample_plan_metrics.py** runs the least-change metrics on the grid, Minnesota, and Wisconsin example plans and outputs scores to a **least_change_outputs** directory
- **map_figures.py** runs our map-generating script to visualize maps of the Minnesota and Wisconin plans and outputs these figures to a **figs** directory
