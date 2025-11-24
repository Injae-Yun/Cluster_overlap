import os
import pickle
import numpy as np
import yaml
from collections import defaultdict
from utils.advanced import plot_cluster_distribution

# -----------------------------------------------------------------
# [중요] 이전 스크립트('main_fix.py')에서 2개의 함수를 가져옵니다.
# (파일 이름이 다르다면 'main_fix' 부분을 수정하세요)
try:
    from main_fix import load_links_and_graph, get_connected_components
except ImportError:
    print("Error: 'main_fix.py' 파일을 찾을 수 없거나, \n"
          "       'load_links_and_graph' 또는 'get_connected_components' 함수가 없습니다.")
    exit()
# -----------------------------------------------------------------

def load_cache(path):
    """ .pkl 파일을 로드하는 헬퍼 함수 """
    with open(path, 'rb') as f:
        return pickle.load(f)

def main_verify():
    """
    저장된 'connected_clusters.pkl' 파일을 분석하여
    불균형 및 커버리지를 검증합니다.
    """
    
    # --- 1. 원본 데이터 및 '타겟 링크' 로드 ---
    
    # [설정] main_fix.py에서 실행한 것과 동일한 설정을 사용해야 합니다.
    MIN_SIZE_FILTER = 100
    CLUSTER_FILE_NAME = 'connected_clusters.pkl'
    
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datapath.yaml'), 'r', encoding='utf-8') as f:
            dp = yaml.safe_load(f)
        # livinglab_path 또는 None을 main_fix.py와 동일하게 설정
        seed_link = dp['common_data']['livinglab_link']
        # seed_link = None 
    except Exception as e:
        print(f"Error loading datapath.yaml: {e}")
        return

    print(f"Loading graph data (seed_link={seed_link})...")
    # link_link_adjacency가 필요하므로 4번째 값까지 받음
    df, _, _, link_adj, _ = load_links_and_graph(seed_link)
    all_links = set(df['LINK_ID'].astype(str))
    print(f"Total links in source file: {len(all_links)}")

    print(f"Finding components with min_size={MIN_SIZE_FILTER}...")
    target_components = get_connected_components(link_adj, all_links, min_size=MIN_SIZE_FILTER)
    
    # '타겟 링크' = min_size 필터를 통과한 모든 컴포넌트의 링크 합집합
    target_links = set()
    for comp in target_components:
        target_links.update(comp)
    
    print(f"Total target links to be clustered: {len(target_links)} (from {len(target_components)} components)")

    # --- 2. 클러스터 결과 로드 ---
    cluster_path = os.path.join(os.path.dirname(__file__), 'Cash', CLUSTER_FILE_NAME)
    print(f"Loading cluster results from {cluster_path}...")
    
    if not os.path.exists(cluster_path):
        print(f"Error: Cluster file not found at {cluster_path}")
        print("Please run main_fix.py first.")
        return
        
    clusters = load_cache(cluster_path) # {id: set_of_links}

    # --- 3. [Task 1] 불균형 검증 (통계) ---
    print("\n--- 1. Cluster Size Verification (Imbalance) ---")
    if not clusters:
        print("No clusters found in file.")
        return
        
    sizes = [len(links) for links in clusters.values()]
    print(f"Total clusters generated: {len(sizes)}")
    if sizes:
        print(f"  Max size: {np.max(sizes)}")
        print(f"  Min size: {np.min(sizes)}")
        print(f"  Mean size: {np.mean(sizes):.2f}")
        print(f"  Median size: {np.median(sizes)}")
        print(f"  Std Dev: {np.std(sizes):.2f}")
        # cluster size N^2 수준 합 비교 (모델 소요시간-복잡도 지표)
        print(f"  Sum of sizes squared: {np.sum(np.array(sizes)**2)}")
        # print("  All sizes:", sizes) # 너무 많으면 주석 처리
    else:
        print("  No cluster sizes to report.")

    # --- 4. [Task 2] 커버리지 검증 ---
    print("\n--- 2. Cluster Coverage Verification ---")
    
    covered_links = set()
    for links in clusters.values():
        covered_links.update(links)
        
    print(f"Total unique links in all clusters: {len(covered_links)}")
    
    # 1. 누락된 링크 (타겟에 있으나, 클러스터 결과에 없는 링크)
    missed_links = target_links - covered_links
    print(f"  [Missed Links] (in target, NOT in clusters): {len(missed_links)}")
    if missed_links:
        print(f"  -> ERROR: {len(missed_links)} links were missed during clustering.")
        # print(f"  Example missed links: {list(missed_links)[:10]}")
        
    # 2. 확장된 링크 (클러스터 결과에 있으나, 타겟에 없는 링크)
    expanded_links = covered_links - target_links
    print(f"  [Expanded Links] (in clusters, NOT in target): {len(expanded_links)}")
    print(f"  -> INFO: This is expected. These links were added by overlap_hops.")

    # 3. 순수 커버리지
    if len(target_links) > 0:
        pure_coverage = len(target_links - missed_links) / len(target_links) * 100
        print(f"  Target Link Coverage: {pure_coverage:.2f}%")
    
    if not missed_links:
        print("\n✅ Verification PASSED: All target links are covered.")
    else:
        print("\n❌ Verification FAILED: Some target links were missed.")
    # jaccard 유사도 계산
    plot_cluster_distribution(clusters,save_path='Cluster_overlap/Cash')


if __name__ == "__main__":
    main_verify()