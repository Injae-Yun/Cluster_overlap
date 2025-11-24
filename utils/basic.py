from dbfread import DBF
import pandas as pd
from scipy.sparse import  save_npz, load_npz
import os
import pickle
import numpy as np
from geopy.distance import geodesic
from shapely.geometry import LineString, Point
import geopandas as gpd
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def load_system_font():
    # matplotlib 한글 폰트 설정
    system = platform.system()
    if system == 'Linux':
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    elif system == 'Windows':
        font_path = 'C:/Windows/Fonts/malgun.ttf'
    elif system == 'Darwin':
        font_path = '/System/Library/Fonts/AppleGothic.ttf'
    else:
        raise RuntimeError('Unsupported OS')
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = [font_prop.get_name(), 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  

def save_cache(base_path,  data):
    with open(base_path, "wb") as f:
        pickle.dump(data, f)

def load_cache(base_path):
    if os.path.exists(base_path):
        with open(base_path, "rb") as f:
            return pickle.load(f)
    return None

def save_sparse_matrix(base_path,  sparse_matrix):
    save_npz(base_path, sparse_matrix)

def load_sparse_matrix(base_path):
    if os.path.exists(base_path):
        return load_npz(base_path)
    return None


def load_moct_link(path, encoding='cp949'):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        print(f"📄 CSV 파일 로딩: {path}")
        df = pd.read_csv(path, low_memory=False)
    elif ext == '.dbf':
        print(f"📁 DBF 파일 로딩: {path}")
        dbf = DBF(path, encoding=encoding)
        df = pd.DataFrame(iter(dbf))
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters

def gps_position_on_link(gps_point, link_info):
    """
    2~3) 링크 위에서의 상대 위치 계산 + T node까지의 거리 추정
    """
    f_point = (link_info['F_LAT'], link_info['F_LON'])
    t_point = (link_info['T_LAT'], link_info['T_LON'])

    dist_to_f = haversine_distance(gps_point[0], gps_point[1], *f_point)
    dist_to_t = haversine_distance(gps_point[0], gps_point[1], *t_point)
    link_length = link_info['LENGTH']

    # GPS 위치가 링크 어디쯤인지 비율 계산
    rel_pos = dist_to_f / (dist_to_f + dist_to_t + 1e-6)  # 방어적 0나눗셈 방지

    # T_NODE까지 남은 거리
    remaining_dist = (1 - rel_pos) * link_length

    return remaining_dist, rel_pos

def gps_position_on_link_geometric(gps_point, link_info):
    """
    링크 위에서의 상대 위치 계산 + T 노드까지의 거리 추정을
    실제 LineString geometry 기반으로 수행

    Parameters:
        gps_point: (lat, lon) tuple
        link_info: dict with at least 'geometry': shapely LineString

    Returns:
        remaining_dist: GPS 위치부터 T_NODE까지 남은 거리 (m)
        rel_pos: 0~1, 링크 상 상대적 위치 (0 = F_NODE, 1 = T_NODE)
    """
    line = link_info['geometry']  # shapely.geometry.LineString

    # 1. 링크 선분에서 GPS 좌표에 가장 가까운 점 찾기
    #projected_point = line.interpolate(line.project(gps_point)) # 계산 결과가 동일하기에 주석처리리

    # 2. 전체 길이 및 현재 지점까지의 길이 계산
    total_length = line.length
    #dist_to_proj = line.project(projected_point) # 시점부터 투영 지점까지의 길이
    dist_to_proj = line.project(gps_point) # 시점부터 투영 지점까지의 길이
    remaining_dist = total_length - dist_to_proj # 남은 거리

    # 상대 위치 비율 (0~1)
    rel_pos = dist_to_proj / total_length if total_length > 0 else 0

    return remaining_dist.values[0][0], rel_pos.values[0][0]

def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    parts = []
    if h > 0:
        parts.append(f"{h}시간")
    if m > 0 or h > 0:
        parts.append(f"{m}분")
    parts.append(f"{s}초")

    return " ".join(parts)