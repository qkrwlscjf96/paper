## 논문 개요
![서강대학교 데이터 사이언스 전공 석사 졸업논문 개요](images/README.png)

## 테스트 데이터 (KAMP 공개 제조 데이터셋 사용) 
 https://www.kamp-ai.kr/aidataList 에서 다운로드 필요
1. 사출성형기
2. 살균기
3. 소성가공
4. 용해탱크

## 프로세스

1. 데이터 로드
2. Feature importance 및 통계 분석
3. EDA 산출물 생성
4. 가중치 적용 전후 모델 성능 비교
5. MLflow에 실험 기록

실행 진입점은 `src/main.py`이며, 실행 대상 데이터셋·단계·모델링 옵션은 코드 내부 `USER_INPUTS`와 환경변수로 제어합니다.

## 디렉터리 설명

- `src/main.py`: 전체 파이프라인 실행 진입점
- `src/utils/func_common.py`: 공통 설정 로드, 데이터 로딩, 파이프라인 보조 함수
- `src/utils/func_eda.py`: EDA 시각화 및 산출물 생성
- `src/utils/func_feat_imp.py`: feature importance 계산 관련 로직
- `src/utils/func_static.py`: 통계 분석 및 이상 탐지 관련 로직
- `src/utils/func_weight.py`: 가중치 데이터 생성 로직
- `src/utils/func_model.py`: 모델 선택 및 하이퍼파라미터 구성
- `src/utils/func_mlflow.py`: MLflow 설정, run 기록, artifact 저장
- `data/`: 원본 실험 데이터셋
- `result/eda/`: EDA 결과물 저장 경로
- `result/mlflow/`: MLflow DB 및 artifact 저장 경로
- `result/logs/`: 실행별 타임스탬프·경과 시간이 포함된 콘솔 로그 저장 경로

## 요구 사항

- Python 3.12 권장
- Docker 선택 사항
- MLflow 사용 시 로컬 Python 환경에 `requirements.txt` 설치 필요

의존성 설치:

Windows (PowerShell):

```powershell
py -3.12 -m pip install -r requirements.txt
```

macOS (Bash/Zsh):

```bash
python3 -m pip install -r requirements.txt
```

## 로컬 실행

기본 실행 명령:

Windows (PowerShell):

```powershell
py -m src.main
```

macOS (Bash/Zsh):

```bash
python3 -m src.main
```

실행 동작은 `src/main.py`의 `USER_INPUTS`에서 제어합니다.

- `data.data_name`, `data.data_names`: 실행할 데이터셋 선택
- `steps.run_pipeline`: 데이터 로드, fold별 분석, sample weight, CV 모델링 전체 실행 여부
- `steps.run_eda`: EDA 실행 여부
- `static.version`: 정적 분석 버전 선택 (`v1`, `v2`)
- `weighting.sample_weight_mul`: INDEX 또는 DATE 통계 조건에 해당하는 행의 학습 가중치

`run_pipeline`과 `run_eda`는 독립적으로 선택할 수 있습니다. EDA를 켜면 필요한 데이터 로드와 전체 데이터 분석은 자동으로 실행됩니다.


## Docker 실행

이미지 빌드:

Windows (PowerShell):

```powershell
docker build -t thesis-model .
```

macOS (Bash/Zsh):

```bash
docker build -t thesis-model .
```

## MLflow + Docker 실행

모델링 결과를 MLflow에 기록하려면 먼저 tracking server를 띄워야 합니다.

### 1. MLflow 저장 폴더 생성

Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force result/mlflow/artifacts
```

macOS (Bash/Zsh):

```bash
mkdir -p result/mlflow/artifacts
```

### 2. MLflow 서버 실행

Windows (PowerShell):

```powershell
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"
py -3.12 -m mlflow server `
  --backend-store-uri "sqlite:///$((Get-Location).Path.Replace('\', '/'))/result/mlflow/mlflow.db" `
  --artifacts-destination "file:///$((Get-Location).Path.Replace('\', '/'))/result/mlflow/artifacts" `
  --serve-artifacts `
  --host 0.0.0.0 `
  --allowed-hosts "127.0.0.1:5001,localhost:5001,host.docker.internal:5001" `
  --port 5001
```

macOS (Bash/Zsh):

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python3 -m mlflow server \
  --backend-store-uri "sqlite:///$PWD/result/mlflow/mlflow.db" \
  --artifacts-destination "file://$PWD/result/mlflow/artifacts" \
  --serve-artifacts \
  --host 0.0.0.0 \
  --allowed-hosts "127.0.0.1:5001,localhost:5001,host.docker.internal:5001" \
  --port 5001
```

브라우저 접속 주소:

- [http://127.0.0.1:5001](http://127.0.0.1:5001)

포트 점유 프로세스 확인 및 종료:

Windows (PowerShell):

```powershell
Get-NetTCPConnection -LocalPort 5001 -ErrorAction SilentlyContinue
Stop-Process -Id <PID>
```

macOS (Bash/Zsh):

```bash
lsof -iTCP:5001 -sTCP:LISTEN -n -P
kill <PID>
```

### 3. Docker 컨테이너에서 학습 실행

전체 데이터셋과 전체 모델 실행 예시:

Windows (PowerShell):

```powershell
docker run --rm `
  -v "${PWD}/result:/app/result" `
  -e EXPERIMENT_PREFIX="[static_v1_260820]" `
  -e DATA_NAMES="all" `
  -e MODEL_NAMES="all" `
  -e STATIC_VERSION="v1" `
  -e RUN_PIPELINE=0 `
  -e RUN_EDA=1 `
  -e MLFLOW_TRACKING_URI="http://host.docker.internal:5001" `
  --name thesis-model `
  thesis-model
```

macOS (Bash/Zsh):

```bash
docker run --rm \
  -e EXPERIMENT_PREFIX="[static_v1_260820]" \
  -e DATA_NAMES="all" \
  -e MODEL_NAMES="all" \
  -e STATIC_VERSION="v1" \
  -e RUN_PIPELINE=0 \
  -e RUN_EDA=1 \
  -e MLFLOW_TRACKING_URI="http://host.docker.internal:5001" \
  --name thesis-model \
  thesis-model
```

## 결과물

- EDA 결과물: `result/eda/<version>/<data_name>/`
- MLflow DB: `result/mlflow/mlflow.db`
- MLflow artifact: `result/mlflow/artifacts/<dataset 이름>/`

실험 이름은 기본적으로 `{EXPERIMENT_PREFIX}--{DATA_NAME}` 형식으로 생성됩니다.  
MLflow run은 `{MODEL_NAME}` 형식으로 모델마다 하나씩 기록됩니다. 각 run의 Metrics 탭에는 `baseline_cv_auc`, `baseline_cv_f1score`, `baseline_cv_precision`, `baseline_cv_recall`, `baseline_cv_accuracy`와 이에 대응하는 `weighted_cv_*` 지표가 함께 표시됩니다. baseline/weighted 구분은 parameter가 아니라 metric 이름에 포함됩니다.

