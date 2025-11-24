from dbfread import DBF
import os
import numpy as np
import geopandas as gpd

from utils.basic import (
    load_moct_link,
    load_system_font)

from utils.advanced import (
    expand_links_cached,
    build_link_adjacency,
    build_link_graph,
    Metis_clustering,
    plot_cluster_distribution)
import yaml

def main(cluster_size=1000,overlap_ratio=0.5,
         seed_link='None',track_visualize=False,):
    # 준비: 상위 datapath.yaml에서 공통 데이터 경로 로드
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datapath.yaml'), 'r', encoding='utf-8') as f:
        dp = yaml.safe_load(f)
    common = dp['common_data']
    dbf_path = common['moct_link_dbf']
    df = load_moct_link(dbf_path)
    #node_dbf_path = "MOCT_NODE.csv"
    #Link_geometry_path = "MOCT_LINK.shp"
    #df_node = load_moct_link(node_dbf_path)
    #gdf = gpd.read_file(Link_geometry_path)

    # 링크 추출
    if seed_link is None:
        cache_dir='Cash/All_link_'+str(cluster_size*2)+'/'
        os.makedirs(cache_dir, exist_ok=True)
        link_list = df['LINK_ID'].tolist()
    elif len(seed_link)==1:
        cache_dir='Cash/'+str(seed_link)+'/'
        os.makedirs(cache_dir, exist_ok=True)
        link_list = expand_links_cached(seed_link, df, adjacency, max_count=10000)
        df = df[df['LINK_ID'].isin(link_list)]
    else:
        link_list= seed_link
        cache_dir='Cash/'+'LivingLab'+'/'
        os.makedirs(cache_dir, exist_ok=True)
        df = df[df['LINK_ID'].isin(link_list)]

    adjacency = build_link_adjacency(df)
    Graph = build_link_graph(df,cache_dir=cache_dir)
    num_cluster=int(len(df)/cluster_size)
    cluster=Metis_clustering(df,adjacency,Graph, k=num_cluster,
                     overlap_ratio=overlap_ratio,cache_dir=cache_dir)
    plot_cluster_distribution(cluster,save_path=cache_dir)
if __name__ == "__main__":
    load_system_font()
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datapath.yaml'), 'r', encoding='utf-8') as f:
        dp = yaml.safe_load(f)
    common = dp['common_data']
    link_list = np.loadtxt(common['livinglab_link'], dtype=str)
   # main(cluster_size=110, overlap_ratio=1,seed_link=link_list, track_visualize=True) # 대체 경로 n개 더 수행


    main(cluster_size=500, overlap_ratio=1,seed_link=None, track_visualize=True) # Total area