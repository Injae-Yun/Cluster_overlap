import numpy as np
import os

import networkx as nx
import networkx as nx
import pymetis
from collections import defaultdict
import pandas as pd
import time

import seaborn as sns
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist, squareform
# 재실행 환경 설정
import folium
from shapely.ops import substring
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map

from collections import defaultdict

from utils.basic import (
    save_cache,
    load_cache,
    save_sparse_matrix,
    load_sparse_matrix,
    gps_position_on_link,
    gps_position_on_link_geometric,
    format_seconds
)
def build_link_adjacency(df):
    from collections import defaultdict
    adjacency = defaultdict(list)
    for _, row in df.iterrows():
        f_node = row['F_NODE']
        link_id = row['LINK_ID']
        t_node = row['T_NODE']
        adjacency[f_node].append((link_id, t_node))
    return adjacency

def expand_links_cached(start_link, df, adjacency, max_count=1000, cache_dir='astar_cache'):
    path=cache_dir+'linklist.pkl'
    # Load cache
    cached = load_cache(path)
    if cached is not None:
        print(f"✅ Loaded link_list from cache for {start_link}")
        return cached
    start_link=str(start_link)
    # Run original function
    link_info = df.set_index('LINK_ID')[['F_NODE', 'T_NODE']]
    visited = set()
    queue = [start_link]
    result = []

    while queue and len(result) < max_count:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        result.append(current)

        try:
            # 현재 링크의 T_NODE
            to_node = link_info.loc[current]['T_NODE']
            # T_NODE를 F_NODE로 갖는 다음 링크 후보들
            next_links = adjacency.get(to_node, [])
            for link_id, _ in next_links:
                if link_id not in visited:
                    queue.append(link_id)
        except:
            continue

    save_cache(path, result)
    print(f"✅ Saved link_list to cache for {start_link}")
    return result

def build_link_graph(df, cache_dir=None):
    path=cache_dir+'link_graph.pkl'
    cached = load_cache(path)
    if cached is not None:
        print(f"✅ Loaded link_Graph from cache for Metis_clustering")
        return cached
    # 노드 간 연결 정보를 인접 리스트로 구성
    G = nx.Graph()
    to_node_dict = defaultdict(list)
    # F_NODE 기반 인접 후보 저장
    for _, row in df.iterrows():
        to_node_dict[row['F_NODE']].append((row['LINK_ID'], row['T_NODE']))
    for _, row in df.iterrows():
        link_i = row['LINK_ID']
        to_node = row['T_NODE']
        for link_j, _ in to_node_dict.get(to_node, []):
            if link_i != link_j:
                G.add_edge(link_i, link_j)
    save_cache(path, G)
    return G

def _expand_single_cluster(args):
    cid, original_links, df, adjacency, target_size, ratio = args
    selected_set = set(original_links)
    needed = int(target_size * ratio)
    candidate_links = set()

    for link in original_links:
        dynamic_max = int(100 * needed / len(original_links))
        expansion = expand_links_cached(link, df, adjacency, max_count=dynamic_max)
        filtered = set(expansion) - selected_set
        candidate_links.update(filtered)

        if len(candidate_links) >= needed:
            break

    final_links = list(selected_set.union(candidate_links))
    return cid, final_links


def expand_links_cached(start_link, df, adjacency, max_count=1000):
    start_link=str(start_link)
    # Run original function
    link_info = df.set_index('LINK_ID')[['F_NODE', 'T_NODE']]
    visited = set()
    queue = [start_link]
    result = []

    while queue and len(result) < max_count:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        result.append(current)

        try:
            # 현재 링크의 T_NODE
            to_node = link_info.loc[current]['T_NODE']
            # T_NODE를 F_NODE로 갖는 다음 링크 후보들
            next_links = adjacency.get(to_node, [])
            for link_id, _ in next_links:
                if link_id not in visited:
                    queue.append(link_id)
        except:
            continue
    return result

def expand_cluster_by_neighbors(cluster_links,degree_dict,
                                df, adjacency, 
                                target_size=1000, ratio=0.5):
    expanded_clusters = {}
    total_clusters = len(cluster_links)
    start_time = time.time()
    for idx, (cid, original_links) in enumerate(cluster_links.items()):
        selected_set = set(original_links)
        needed = int(target_size * ratio) #2000 -> 1000
        candidate_links = set()

        for link in original_links:
            dynamic_max = int(100 * needed / len(original_links))
            expansion = expand_links_cached(link, df, adjacency, max_count=dynamic_max)
            filtered = set(expansion) - selected_set  # 원래 없던 링크만 남기기
            candidate_links.update(filtered)

            if len(candidate_links) >= needed: #for select more sociable links
                break
        # 최종 클러스터: 원래 링크 + 확장된 이웃
        #sorted_neighbors = sorted(candidate_links, key=lambda x: degree_dict.get(x, 0), reverse=True)
        #top_neighbors = list(sorted_neighbors)[:needed] 
        #final_links = list(selected_set.union(top_neighbors))
        final_links = list(selected_set.union(candidate_links))
        expanded_clusters[cid] = final_links #[:target_size]
        elapsed = time.time() - start_time
        percent = (idx + 1) / total_clusters
        eta = elapsed / percent * (1 - percent)
        print(f"✅ [{idx+1}/{total_clusters}] Cluster {cid} 완료 - {percent*100:.2f}% 진행됨 - 예상 남은 시간: {eta:.1f}초")
    
    total_time = time.time() - start_time
    print(f"\n✅ 전체 클러스터 확장 완료! 총 소요 시간: {total_time:.1f}초")
    return expanded_clusters


def Metis_clustering(df,adjacency,G, k=500,overlap_ratio=1, cache_dir=None):
    """
    Metis 클러스터링을 사용하여 그래프를 클러스터링합니다.
    """
    path=cache_dir+'expanded_clusters.pkl'
    cached = load_cache(path)
    if cached is not None:
        print(f"✅ Loaded expanded_clusters from cache for METIS")
        return cached
    
    # 1. 노드 인덱싱
    node_list = list(G.nodes())
    node_index = {node: i for i, node in enumerate(node_list)}

    # 2. METIS 입력 포맷 (connect list)
    connect = [[] for _ in range(len(node_list))]
    for u, v in G.edges():
        connect[node_index[u]].append(node_index[v])
        connect[node_index[v]].append(node_index[u])

    # 3. METIS 클러스터링
    _, parts = pymetis.part_graph(k, adjacency=connect)

    # 4. 클러스터 결과 정리
    cluster_dict = defaultdict(list)
    for node, part in zip(node_list, parts):
        cluster_dict[part].append(node)

    # 5. Overlap 확장 (이웃 노드 중 일부 추가)
    target_size =len(cluster_dict[0])*(1+overlap_ratio)
    ratio = overlap_ratio/(1+overlap_ratio)
    degree_dict = dict(G.degree())  # 링크별 연결도

    expanded_clusters = expand_cluster_by_neighbors(
        cluster_dict, degree_dict, df, adjacency, target_size=target_size, ratio=ratio)


    print(f"✅ METIS clustering 완료 (k={k}, overlap_ratio={overlap_ratio})")
    save_cache(path, expanded_clusters)

    return expanded_clusters



def plot_cluster_distribution(expanded_clusters,save_path=None): 
    cluster_ids = list(expanded_clusters.keys())
    jaccard_matrix = np.zeros((len(cluster_ids), len(cluster_ids)))

    for i in tqdm(range(len(cluster_ids))):
        set_i = set(expanded_clusters[cluster_ids[i]])
        for j in range(i+1, len(cluster_ids)):
            set_j = set(expanded_clusters[cluster_ids[j]])
            inter = len(set_i & set_j)
            union = len(set_i | set_j)
            jaccard_matrix[i, j] = inter / union if union != 0 else 0
            jaccard_matrix[j, i] = jaccard_matrix[i, j]

    # 2. 거리 기반 정렬
    # (주의) Jaccard similarity -> Distance로 변환
    distance_matrix = 1 - jaccard_matrix

    # (2-1) Cosine Distance를 쓰고 싶으면 이 부분만 바꿔
    #distance_matrix = squareform(pdist(jaccard_matrix, metric='cosine'))
    np.fill_diagonal(distance_matrix, 0)

    # (2-2) Hierarchical clustering으로 leaf order 얻기
    linkage_matrix = linkage(squareform(distance_matrix), method='average')
    order = leaves_list(linkage_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(jaccard_matrix[order][:, order], cmap="coolwarm") 
    plt.title(" Jaccard similarity (Overlap) - sorted")
    plt.xlabel("Cluster ID (sorted)")
    plt.ylabel("Cluster ID (sorted)")
    plt.tight_layout()
    sorted_path = os.path.join(save_path, "jaccard_sorted_heatmap.png")
    plt.savefig(sorted_path)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(jaccard_matrix, cmap="coolwarm") 
    plt.title("Jaccard similarity (Overlap)")
    plt.xlabel("Cluster ID")
    plt.ylabel("Cluster ID")
    plt.tight_layout()
    sorted_path = os.path.join(save_path, "jaccard_heatmap.png")
    plt.savefig(sorted_path)
    plt.close()