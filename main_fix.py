import os
import numpy as np
import geopandas as gpd
import networkx as nx
import yaml
import random # 시드 선정에 필요
from collections import deque, defaultdict
from utils.basic import save_cache
from utils.advanced import Metis_clustering, build_link_graph, build_link_adjacency as build_node_link_adjacency

# --- build_link_link_adjacency (변경 없음) ---
def build_link_link_adjacency(df):
    node_to_links = defaultdict(list)
    for _, row in df.iterrows():
        node_to_links[str(row['F_NODE'])].append(str(row['LINK_ID']))
        node_to_links[str(row['T_NODE'])].append(str(row['LINK_ID']))
    link_adjacency = defaultdict(set)
    for _, row in df.iterrows():
        link_id = str(row['LINK_ID'])
        f_node = str(row['F_NODE'])
        t_node = str(row['T_NODE'])
        neighbors = set(node_to_links[f_node] + node_to_links[t_node])
        neighbors.discard(link_id)
        link_adjacency[link_id] = neighbors
    return link_adjacency

# --- load_links_and_graph (변경 없음) ---
def load_links_and_graph(seed_link=None):
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datapath.yaml'), 'r', encoding='utf-8') as f:
        dp = yaml.safe_load(f)
    common = dp['common_data']
    df = gpd.read_file(common['moct_link_shp'])
    if seed_link is not None:
        if isinstance(seed_link, str):
            link_list = np.loadtxt(seed_link, dtype=str)
        else:
            link_list = seed_link
        df = df[df['LINK_ID'].astype(str).isin(link_list)]
    linkid_to_nodes = {str(row['LINK_ID']): (str(row['F_NODE']), str(row['T_NODE'])) for _, row in df.iterrows()}
    nodes_to_linkid = {(str(row['F_NODE']), str(row['T_NODE'])): str(row['LINK_ID']) for _, row in df.iterrows()}
    link_link_adjacency = build_link_link_adjacency(df)
    node_link_adjacency = build_node_link_adjacency(df)
    return df, linkid_to_nodes, nodes_to_linkid, link_link_adjacency, node_link_adjacency

# --- get_connected_components (원복: min_size 필터 유지) ---
def get_connected_components(link_adjacency, all_links, min_size=100):
    unvisited = set(all_links)
    components = []
    while unvisited:
        seed = next(iter(unvisited))
        queue = deque([seed])
        visited = set([seed])
        component = []
        while queue:
            curr = queue.popleft()
            component.append(curr)
            for nbr in link_adjacency[curr]:
                if nbr in unvisited and nbr not in visited:
                    queue.append(nbr)
                    visited.add(nbr)
        components.append(set(component))
        unvisited -= set(component)
        
    print(f"[INFO] Disconnected component sizes (min_size={min_size}):")
    filtered = []
    for idx, comp in enumerate(components):
        print(f"  Component {idx}: {len(comp)} links")
        if len(comp) > min_size:
            filtered.append(comp)
    return filtered

# --- assign_clusters_to_components (수정본) ---
def assign_clusters_to_components(components, total_clusters):
    sizes = [len(comp) for comp in components]
    total = sum(sizes)
    if total == 0:
        return []
    
    raw_alloc = [s / total * total_clusters for s in sizes]
    alloc = [max(1, int(round(x))) for x in raw_alloc]
    diff = total_clusters - sum(alloc)
    
    remainders = [(raw_alloc[i] - np.floor(raw_alloc[i]), i) for i in range(len(raw_alloc))]

    if diff > 0:
        remainders.sort(key=lambda x: x[0], reverse=True)
        for i in range(diff):
            idx_to_add = remainders[i % len(remainders)][1] # 순환하며 추가
            alloc[idx_to_add] += 1
            
    elif diff < 0:
        remainders.sort(key=lambda x: x[0])
        to_remove = abs(diff)
        removed_count = 0
        
        for i in range(len(remainders)):
            if removed_count == to_remove:
                break
            idx_to_remove = remainders[i][1]
            if alloc[idx_to_remove] > 1:
                alloc[idx_to_remove] -= 1
                removed_count += 1
        
        final_diff = total_clusters - sum(alloc)
        if final_diff != 0:
             print(f"[WARNING] Cannot meet target {total_clusters}. "
                   f"Allocated {sum(alloc)} clusters (min 1 per component).")

    return alloc

# --- [신규] 1. 시드 선정 (Farthest-First Traversal) ---
def select_seeds(graph: nx.Graph, k: int):
    """
    Farthest-First Traversal 휴리스틱을 사용해 k개의 시드를 선택합니다.
    """
    nodes = list(graph.nodes)
    if not nodes:
        return []
    if k >= len(nodes):
        return nodes # k가 노드 수보다 많으면 모든 노드를 시드로 반환
    
    # 1. 첫 번째 시드는 랜덤 선택
    seeds = [random.choice(nodes)]
    
    # 각 노드에서 *가장 가까운* 시드까지의 거리를 저장
    min_dists = {}
    for node in nodes:
        try:
            min_dists[node] = nx.shortest_path_length(graph, source=seeds[0], target=node)
        except nx.NetworkXNoPath:
            min_dists[node] = float('inf') # 도달 불가능 (컴포넌트 내에선 없어야 함)

    while len(seeds) < k:
        # 2. 현재 시드 집합에서 *가장 먼* 노드를 다음 시드로 선택
        farthest_node = max(min_dists, key=min_dists.get)
        if min_dists[farthest_node] == float('inf'):
            # 그래프가 분리된 경우 (이론상 component에선 발생 안 함)
            # 남은 노드 중 하나를 랜덤 선택
            remaining_nodes = list(set(nodes) - set(seeds))
            if not remaining_nodes: break
            farthest_node = random.choice(remaining_nodes)

        seeds.append(farthest_node)
        
        # 3. min_dists 딕셔너리 업데이트
        # 방금 추가된 시드(farthest_node)를 기준으로 거리 계산
        nodes_to_update = list(min_dists.keys()) # del 도중 순회 오류 방지
        for node in nodes_to_update:
            try:
                d = nx.shortest_path_length(graph, source=farthest_node, target=node)
                min_dists[node] = min(min_dists[node], d)
            except nx.NetworkXNoPath:
                continue # 기존 최소 거리 유지
        
        # 4. 선택된 시드는 후보에서 제거
        del min_dists[farthest_node] 
            
    return seeds

# --- [신규] 2. 핵심 파티션 성장 (Multi-Source BFS) ---
# --- [신규] 2. 핵심 파티션 성장 (Multi-Source BFS) ---
# [수정됨] 크기 균형을 고려하는 버전
def grow_core_partitions(component_links: set, link_adjacency: dict, seeds: list, leeway_factor=0.1):
    """
    시드로부터 동시 BFS를 수행하되, 파티션 크기가 목표치를 넘으면 성장을 멈춰 균형을 맞춘다.
    """
    k = len(seeds)
    if k == 0:
        return []
        
    # 1. 목표 크기 설정 (전체 크기 / k) + 여유분
    target_size = len(component_links) / k
    max_size = int(target_size * (1.0 + leeway_factor)) # 예: 10% 여유
    print(f"    [SRG-B] Target size: ~{int(target_size)}, Max size: {max_size}")

    partitions = {i: set() for i in range(k)}
    assigned_nodes = set()
    queue = deque()
    
    # 큐 활성화 상태 (파티션이 max_size에 도달하면 False로)
    cluster_active = {i: True for i in range(k)}

    # 큐 초기화
    for i, seed in enumerate(seeds):
        if seed in component_links and seed not in assigned_nodes:
            partitions[i].add(seed)
            assigned_nodes.add(seed)
            queue.append((seed, i)) # (노드, 파티션 ID)
        else:
            cluster_active[i] = False # 유효하지 않은 시드

    # 2. 1단계: 크기 제한이 있는 동시 BFS
    while queue:
        curr, cluster_id = queue.popleft()
        
        # 이 파티션이 비활성화(꽉 참) 상태면, 큐에 남아있던 작업은 스킵
        if not cluster_active[cluster_id]:
            continue

        for nbr in link_adjacency[curr]:
            # 1. 이웃이 이 컴포넌트에 속하고, 2. 아직 할당되지 않았어야 함
            if nbr in component_links and nbr not in assigned_nodes:
                
                # 3. 크기 제한 체크
                if len(partitions[cluster_id]) < max_size:
                    assigned_nodes.add(nbr)
                    partitions[cluster_id].add(nbr)
                    queue.append((nbr, cluster_id))
                else:
                    # 이 파티션은 꽉 찼으므로 비활성화
                    cluster_active[cluster_id] = False
                    # 큐에 더 이상 이 cluster_id로 탐색을 넣지 않음
                    # (참고: 이미 큐에 들어간 작업은 위에서 스킵됨)

    # 3. 2단계: 미할당 노드 처리 (Leftovers)
    # BFS가 끝난 후에도 할당되지 않은 '고아' 노드들
    unassigned_nodes = component_links - assigned_nodes
    print(f"    [SRG-B] {len(unassigned_nodes)} leftover links to assign.")
    
    if unassigned_nodes:
        # 각 파티션의 '경계' 노드를 찾음 (미할당 노드와 인접한 노드)
        border_nodes = {i: set() for i in range(k)}
        for node in unassigned_nodes:
            for nbr in link_adjacency[node]:
                if nbr in assigned_nodes:
                    # nbr가 속한 파티션을 찾아야 함
                    for pid, p_nodes in partitions.items():
                        if nbr in p_nodes:
                            border_nodes[pid].add(node) # 'node'는 경계에 있음
                            break
        
        # BFS 큐를 다시 사용 (이번엔 거리 1부터 탐색)
        queue_leftover = deque()
        for pid, nodes in border_nodes.items():
            for node in nodes:
                if node in unassigned_nodes: # 아직 할당 안 된 고아 노드만 큐에 추가
                    queue_leftover.append((node, pid))
                    assigned_nodes.add(node) # 큐에 넣는 순간 할당된 것으로 간주
                    partitions[pid].add(node)

        # 고아 노드들에 대한 BFS (크기 제한 없음, 가장 가까운 곳에 붙임)
        while queue_leftover:
            curr, cluster_id = queue_leftover.popleft()
            for nbr in link_adjacency[curr]:
                if nbr in unassigned_nodes and nbr not in assigned_nodes:
                    assigned_nodes.add(nbr)
                    partitions[cluster_id].add(nbr)
                    queue_leftover.append((nbr, cluster_id))

    return list(partitions.values())
# --- [신규] 2.5. 작은 파티션 병합 (Post-processing) ---
def merge_small_partitions(core_partitions: list, 
                           link_adjacency: dict, 
                           component_links: set, 
                           target_size: float, 
                           leeway_factor=0.15):
    """
    핵심 파티션 생성 후, min_size 임계값보다 작은 파티션을
    가장 경계를 많이 공유하는 이웃 파티션에 병합합니다.
    """
    
    # 1. 최소 크기 임계값 설정
    # (leeway_factor를 더 엄격하게 적용하거나, 0.5 같은 비율을 곱할 수 있습니다)
    min_size_threshold = int(target_size * (1.0 - leeway_factor))
    # 예: target=105, leeway=0.15 -> min=89. 18은 무조건 병합.
    
    print(f"    [Merge] Merging clusters smaller than {min_size_threshold} (Target: {target_size:.0f})...")
    
    # 2. 리스트(list[set])를 딕셔너리(dict{id: set})로 변환 (병합에 용이)
    partitions = {i: p for i, p in enumerate(core_partitions) if p}
    if not partitions:
        return []

    # 3. {노드: 파티션ID} 역방향 맵 생성 (이웃 탐색용)
    node_to_pid = {}
    for pid, nodes in partitions.items():
        for node in nodes:
            # component_links에 속한 노드만 매핑 (확장된 노드 제외)
            if node in component_links:
                node_to_pid[node] = pid

    # 4. 더 이상 병합할 것이 없을 때까지 반복
    merged_something = True
    while merged_something:
        merged_something = False
        
        # 5. 임계값 미만의 가장 작은 파티션 검색
        smallest_pid = -1
        smallest_size = float('inf')
        
        for pid, nodes in partitions.items():
            # '핵심' 크기(comp 내) 기준으로 판단
            core_size = len(nodes.intersection(component_links))
            if core_size < min_size_threshold and core_size < smallest_size:
                smallest_size = core_size
                smallest_pid = pid
        
        # 6. 종료 조건: 병합할 작은 파티션이 없음
        if smallest_pid == -1:
            break
            
        merged_something = True # 병합 대상 찾음
        p_small = partitions[smallest_pid]
        print(f"    [Merge] Found small cluster {smallest_pid} (core size {smallest_size})...")

        # 7. 병합할 이웃 찾기 (경계가 가장 큰)
        neighbor_boundary_count = defaultdict(int)
        for node in p_small:
            if node not in component_links: continue # 핵심 노드만 검사
            
            for nbr in link_adjacency[node]:
                # 이웃이 다른 파티션에 속해 있다면
                if nbr in node_to_pid and node_to_pid[nbr] != smallest_pid:
                    neighbor_pid = node_to_pid[nbr]
                    neighbor_boundary_count[neighbor_pid] += 1
        
        # 8. 병합 대상 이웃 확정
        if not neighbor_boundary_count:
            print(f"    [Merge] WARNING: Small cluster {smallest_pid} is isolated. Cannot merge.")
            del partitions[smallest_pid] # 무한 루프 방지를 위해 제거
            continue
            
        # 가장 경계를 많이 공유하는 이웃
        best_neighbor_pid = max(neighbor_boundary_count, key=neighbor_boundary_count.get)
        
        # 9. 병합 수행
        print(f"    [Merge] ...merging into cluster {best_neighbor_pid} (boundary links: {neighbor_boundary_count[best_neighbor_pid]}).")
        
        # (1) 노드를 이웃 파티션으로 이동
        partitions[best_neighbor_pid].update(p_small)
        
        # (2) 역방향 맵 업데이트
        for node in p_small:
            if node in component_links:
                node_to_pid[node] = best_neighbor_pid
                
        # (3) 원본 파티션 삭제
        del partitions[smallest_pid]

    print(f"    [Merge] Merge complete. Final cluster count: {len(partitions)}")
    return list(partitions.values())

# --- [신규] 3. 오버랩 확장 (d-Hop Expansion) ---
def expand_partitions_with_overlap(core_partitions: list, link_adjacency: dict, overlap_hops: int):
    """
    각 핵심 파티션을 d-hop 만큼 확장하여 오버랩을 생성합니다.
    """
    if overlap_hops == 0:
        return core_partitions # 확장이 0이면 원본 반환

    expanded_partitions = []
    for core_partition in core_partitions:
        if not core_partition: # 비어있는 파티션 스킵
            continue
        
        current_expansion = set(core_partition) # 최종 확장 영역
        border = set(core_partition)          # 현재 홉에서 탐색을 시작할 경계
        
        for _ in range(overlap_hops):
            new_border = set()
            for node in border:
                for nbr in link_adjacency[node]:
                    if nbr not in current_expansion:
                        # 이 nbr은 이번 홉에서 새로 추가된 노드임
                        new_border.add(nbr)
            
            if not new_border: # 더 이상 확장할 노드가 없으면 중단
                break
                
            current_expansion.update(new_border)
            border = new_border # 다음 홉은 방금 추가된 new_border에서 시작
            
        expanded_partitions.append(current_expansion)
        
    return expanded_partitions


# --- main (수정됨) ---
def main(num_cluster=40, overlap_step=1, seed_link=None):
    df, linkid_to_nodes, nodes_to_linkid, link_link_adjacency, node_link_adjacency = load_links_and_graph(seed_link)
    if seed_link is None:
        cache_dir='Cash/All_link_'+str(num_cluster)+'/'
        os.makedirs(cache_dir, exist_ok=True)
        link_list = df['LINK_ID'].tolist()
    else:
        link_list= seed_link
        cache_dir='Cash/'+'LivingLab'+'/'
        os.makedirs(cache_dir, exist_ok=True)

    all_links = set(linkid_to_nodes.keys())
    
    # 1. 필터가 적용된 컴포넌트 가져오기
    components = get_connected_components(link_link_adjacency, all_links, min_size=100)
    
    print("[INFO] Large components for clustering:")
    for idx, comp in enumerate(components):
        print(f"  Component {idx}: {len(comp)} links")
        
    # 2. 클러스터 개수 할당
    alloc = assign_clusters_to_components(components, num_cluster)
    print(f"[INFO] Cluster allocation per component: {alloc}")
    
    clusters = {}
    cluster_id = 0
    
    # 3. 필터링된 각 컴포넌트에 대해 SRG 클러스터링 수행
    for comp, n_clu in zip(components, alloc):
        if n_clu == 0:
            continue
            
        print(f"  Processing component (size {len(comp)}) -> {n_clu} clusters...")
        sub_df = df[df['LINK_ID'].astype(str).isin(comp)]
        
        all_nodes_in_comp = set(sub_df['F_NODE'].astype(str)) | set(sub_df['T_NODE'].astype(str))
        sub_node_link_adj = {k: node_link_adjacency[k] for k in all_nodes_in_comp if k in node_link_adjacency}
        
        G = build_link_graph(sub_df,cache_dir=cache_dir)
        
        # --- [대체됨] Metis_clustering 호출 대신 SRG 방식 사용 ---
        # [설정] leeway_factor 정의 (병합 시에도 사용)
        target_size = len(comp) / n_clu        # 1. 시드 선정
        leeway_factor = 0.15
        print(f"    [SRG] Selecting {n_clu} seeds...")
        seeds = select_seeds(G, n_clu)
        
        # 2. 핵심 파티션 성장
        print(f"    [SRG] Growing {len(seeds)} core partitions...")
        # 성장(grow)은 전체 adj를 사용하되, comp 셋 내부로 제한
        core_partitions = grow_core_partitions(
            comp, 
            link_link_adjacency, 
            seeds,
            leeway_factor=leeway_factor # 15% 정도의 크기 불균형 허용
        )
        # --- [신규 삽입] 2.5. 작은 파티션 병합 ---
        # `grow_core_partitions_balanced` 직후, `expand_...` 직전
        core_partitions = merge_small_partitions(
            core_partitions,
            link_link_adjacency,
            comp, # '핵심' 노드 셋
            target_size,
            leeway_factor=leeway_factor*2 #target size 1-0.2도 안 되는 경우는 병합 #약 50 개 수준
        )
        # 3. 오버랩 확장
        # overlap_step=1 -> 1-hop, overlap_step=2 -> 2-hop으로 해석
        overlap_hops = int(round(overlap_step))
        print(f"    [SRG] Expanding partitions with {overlap_hops}-hop overlap...")
        # 확장(expand)은 전체 adj를 사용하며, comp 외부로 나갈 수 있음 (오버랩)
        expanded_link_sets = expand_partitions_with_overlap(
            core_partitions, 
            link_link_adjacency, 
            overlap_hops
        )

        
        # ---------------------------------------------------
        
        # 결과를 {id: set} 딕셔너리로 변환
        expanded_clusters = {i: links for i, links in enumerate(expanded_link_sets) if links}

        # 최종 클러스터 딕셔너리에 추가
        for cid, links in expanded_clusters.items():
            # cid는 컴포넌트 내의 로컬 ID (0, 1, ... n_clu-1)
            # cluster_id는 전역 ID (0, 1, ... num_cluster-1)
            clusters[cluster_id] = links
            cluster_id += 1
            
    out_path = os.path.join(os.path.dirname(__file__), 'Cash', 'connected_clusters.pkl')
    save_cache(out_path, clusters)
    print(f"✅ {len(clusters)} connected clusters saved to {out_path}")
    print(f"  (Target: {num_cluster}, Allocated: {sum(alloc)})") # 실제 생성된 클러스터 수 로깅
    print()

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datapath.yaml'), 'r', encoding='utf-8') as f:
        dp = yaml.safe_load(f)
    livinglab_path = dp['common_data']['livinglab_link']
    
    main(num_cluster=50, overlap_step=2, seed_link=livinglab_path)
    #main(num_cluster=1000, overlap_ratio=0.8, seed_link=None)