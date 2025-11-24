## Cluster_overlap 프로젝트

간단한 설명
- 이 레포지토리는 도로/링크 정보를 기반으로 Metis 클러스터링을 수행하여 클러스터 분포 및 클러스터 간 중첩(유사도)을 분석합니다.
- 주요 실행 엔트리: `main.py` — `datapath.yaml`에서 경로를 읽어와 링크 DB를 로드하고 클러스터를 생성합니다.

## 필수 데이터 (중요)
- 이 저장소에는 `datapath.yaml` 파일이 포함되어 있지 않습니다.
- 코드가 기대하는 최소한의 ITS 기본 정보 (버전: 2024-02)를 별도로 다운로드하여 사용해야 합니다.

권장 데이터 파일
- MOCT 링크 정보 (예: `MOCT_LINK.dbf` 또는 `MOCT_LINK.csv`) — `main.py`에서 `common_data.moct_link_dbf`로 참조
- LivingLab 링크 목록 (예: 텍스트 파일 또는 CSV) — `main.py`의 `common_data.livinglab_link`로 참조

※ 파일명과 포맷은 프로젝트 내에서 유연하게 처리됩니다. `utils/basic.py`의 `load_moct_link` 함수는 `.csv` 또는 `.dbf` 확장자를 자동으로 처리합니다.

## datapath.yaml 예제
아래 템플릿을 프로젝트 루트(또는 `main.py`의 상위 경로) 근처에 `datapath.yaml`로 만들어 주세요.

```yaml
common_data:
  # MOCT 링크 파일 경로 (.dbf 또는 .csv)
  moct_link_dbf: "/absolute/or/relative/path/to/MOCT_LINK.dbf"

  # LivingLab (혹은 seed link list) 경로 - np.loadtxt로 읽습니다 (한 행에 하나의 LINK_ID)
  livinglab_link: "/absolute/or/relative/path/to/livinglab_link.txt"

  # 필요하면 다른 데이터 경로를 여기에 추가하세요.
```

경로 지정 팁
- 상대 경로를 쓸 경우 `datapath.yaml`이 프로젝트의 루트(또는 `main.py`의 부모 폴더)에서 참조되는 방식에 맞추어 지정하세요.

## ITS 기본 정보(2024-02) 다운로드 안내
- 'ITS 기본 정보 (2024-02)'은 공공기관 또는 ITS 관련 포털에서 배포되는 도로·링크(노드) 데이터입니다. 배포처 예시:
  - 국토교통부 / 지역교통 포털, 또는 ITS 관련 데이터 포털
  - 지자체 또는 ITS 사업자에서 제공하는 공개 데이터 페이지

다운로드 후 확인해야 할 항목
- 링크 파일에 `LINK_ID`, `F_NODE`, `T_NODE`, `F_LAT`, `F_LON`, `T_LAT`, `T_LON`, `LENGTH` 등의 컬럼이 포함되어 있어야 합니다. `main.py`와 `utils`들이 이 컬럼들을 이용합니다.
- 만약 DBF가 아닌 CSV로 제공된다면, `datapath.yaml`에서 `.csv` 파일 경로를 지정하면 됩니다.

권장 작업 흐름
1. ITS 데이터 포털에서 'ITS 기본 정보 (2024-02)'를 다운로드합니다.
2. 필요한 파일(MOCT 링크 DBF/CSV, livinglab link list)을 프로젝트 내 적절한 디렉토리에 놓습니다.
3. 위 예제대로 `datapath.yaml`을 작성합니다.
4. (선택) `Cash/` 폴더는 캐시와 산출물을 저장하므로 쓰기 권한이 필요합니다. 필요하면 해당 폴더를 미리 만들어 두세요.

## 환경 및 의존성
- Python 3.8+ 권장
- 필요한 패키지는 `requirements.txt`에 목록화되어 있습니다. (아래 참조)

설치 예시
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

## 실행 방법
- LivingLab 시드로 여러 번 실행하려면 `main.py`의 `__main__` 섹션을 수정하거나 주석을 참고하세요.

기본 실행 (전체 영역)
```bash
python main.py
```

특정 옵션으로 실행하려면 `main()`을 직접 호출하거나 스크립트를 수정하세요. `main.py` 내부에서 `cluster_size`, `overlap_ratio`, `seed_link` 등을 인자로 전달할 수 있습니다.

## 파일 설명(주요)
- `main.py` : 워크플로우 엔트리. `datapath.yaml`에서 경로를 읽고 클러스터링 실행.
- `utils/basic.py` : 파일 로드, 거리 계산 등 기본 유틸 함수.
- `utils/advanced.py` : 그래프 생성, Metis 클러스터링, 클러스터 확장 및 시각화.

## 주의사항
- `pymetis` 설치는 플랫폼에 따라 추가 빌드 도구(예: C 컴파일러)가 필요할 수 있습니다. 설치 실패 시 시스템 패키지(예: build-essential)를 설치하세요.
- `geopandas` 및 `shapely`는 바이너리 의존성이 있어 설치가 번거로울 수 있습니다. conda 환경 사용을 권장합니다.

## 추가 도움
- 필요한 부분(예: datapath.yaml 예제 경로, ITS 데이터 다운로드 링크 등)을 제공해 주시면 README를 더 구체적으로 업데이트해 드리겠습니다.

--
작업한 파일: 이 README는 프로젝트 루트의 `README.md` 입니다.
