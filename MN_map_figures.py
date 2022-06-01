import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt

def draw_graph(G, plan_assignment, unit_df, division_df, fig_name, geo_id ='GEOID10'):
    cdict = {G.nodes[i][geo_id]:plan_assignment[i] for i in plan_assignment.keys()}
    unit_df['color'] = unit_df.apply(lambda x: cdict[x[geo_id]], axis=1)
    fig,ax = plt.subplots()
    division_df.geometry.boundary.plot(color=None,edgecolor='k',linewidth = 1,ax=ax)
    unit_df.plot(column='color',ax = ax, cmap = 'tab20')
    ax.set_axis_off()
    plt.savefig(fig_name, dpi = 300)
    plt.close()

color_list_example = [[0.72156862745098, 0.580392156862745, 0.713725490196078], 
    [0.862745098039216, 0.713725490196078, 0.274509803921569], 
    [0.8, 0.392156862745098, 0.325490196078431], 
    [0.682352941176471, 0.549019607843137, 0.380392156862745],
    [0.39215686274509803, 0.5843137254901961, 0.9294117647058824], 
    # [0.352941176470588, 0.694117647058824, 0.803921568627451], 
    [0.329411764705882, 0.47843137254902, 0.250980392156863], 
    [0.701960784313725, 0.933333333333333, 0.756862745098039],
    # [0.470588235294118, 0.419607843137255, 0.607843137254902],
    [0.5450980392156862, 0.2784313725490196, 0.5372549019607843],
    [0.701960784313725, 0.733333333333333, 0.823529411764706],
    [0.92156862745098, 0.909803921568627, 0.776470588235294]]

# RGB(100,149,237),RGB(186,85,211)
def draw_graph_w_division(district_df, assignment_col, color_col, fig_name, color_type = 'cmap',cmap = 'tab20', vmin= None, vmax = None, colorbar = False, color_list = color_list_example, district_labels = True, division_df = None, div_lw = 6, div_color = 'black',dist_outline = False, dist_lw = 6, dist_color = 'black', inset = None, area_label_min = 1e+9, dpi = 500):
    fig,ax = plt.subplots()
    if division_df is not None:
        division_df.geometry.boundary.plot(color=None,edgecolor=div_color,linewidth = div_lw,ax=ax)
    if dist_outline:
        dists_dissolve = district_df.dissolve(by = color_col).reset_index()
        dists_dissolve.geometry.boundary.plot(color=None,edgecolor=dist_color,linewidth = dist_lw,ax=ax)
    if color_type == 'cmap':
        district_df.plot(column=color_col,ax = ax, cmap = cmap, vmin=  vmin, vmax = vmax, legend=colorbar, legend_kwds={'label': color_col})
    elif color_type == 'list':
        district_df.plot(color = [color_list[i] for i in list(district_df[color_col])], ax=ax)
    if district_labels:
        for i in range(len(district_df)):
            if district_df['geometry'].area[i] >= area_label_min:
                plt.annotate(district_df[assignment_col][i],(district_df["geometry"].centroid[i].x, district_df["geometry"].centroid[i].y), c='black', fontsize = 5, zorder = 3, ha='center', va='center')    
    if inset:
        ax.set_xlim(inset[0])
        ax.set_ylim(inset[1])
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(fig_name,dpi = dpi)
    plt.close()    


geoid = 'VTD'
mn_shp = gpd.read_file('./input_data/mn20_shapefile/')
least_change_mn_plans = pd.read_csv('./input_data/least_change_plans_new.csv', dtype = {geoid: 'str'})
subdiv_splits_mn_plans = pd.read_csv('./input_data/splits_MN20_plans.csv', dtype = {geoid: 'str'})
mn_plan_merge = mn_shp.merge(least_change_mn_plans.rename(columns = {plan:'LC_'+plan for plan in [col for col in least_change_mn_plans.columns if 'PLAN' in col.upper()]}), how = 'left', left_on = 'VTD', right_on = 'VTD')
mn_plan_merge = mn_plan_merge.merge(subdiv_splits_mn_plans.rename(columns = {plan:'SPLITS_'+plan for plan in [col for col in subdiv_splits_mn_plans.columns if col!=geoid]}), how = 'left', left_on = 'VTD', right_on = 'VTD')
county_shp = mn_shp.dissolve(by = 'CNTY_FIPS').reset_index()


figsdir = './figs/'
os.makedirs(os.path.dirname(figsdir), exist_ok=True)

for plan in ['Enacted']+[col for col in mn_plan_merge.columns if 'SPLITS_' in col.upper()]:
    draw_graph_w_division(mn_plan_merge, plan, plan, figsdir+'MN_'+plan+'_map_full.png', color_type = 'list',cmap = 'tab20', district_labels = False, division_df = county_shp, div_lw = 2, div_color = 'black',inset = None, dpi = 500)
    draw_graph_w_division(mn_plan_merge, plan, plan, figsdir+'MN_'+plan+'_map_inset.png', color_type = 'list',cmap = 'tab20', district_labels = False, division_df = county_shp, div_lw = 6, div_color = 'black', inset = ((428316,530062),(4940966,5033176)), dpi = 500)

for plan in ['cong_assig']+[col for col in mn_plan_merge.columns if 'LC_PLAN' in col.upper()]:
    draw_graph_w_division(mn_plan_merge, plan, plan, figsdir+'MN_'+plan+'_map_full.png', color_type = 'list',cmap = 'tab20', district_labels = False, division_df = county_shp, inset = None, dpi = 500,div_lw = 0.5, div_color = 'black',dist_outline = True, dist_lw = 2, dist_color = 'black')
    draw_graph_w_division(mn_plan_merge, plan, plan, figsdir+'MN_'+plan+'_map_inset.png', color_type = 'list',cmap = 'tab20', district_labels = False, division_df = county_shp, inset = ((428316,530062),(4940966,5033176)), dpi = 500,div_lw = 1, div_color = 'black',dist_outline = True, dist_lw = 4, dist_color = 'black')


