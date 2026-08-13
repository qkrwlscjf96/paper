## 논문 개요
![서강대학교 데이터 사이언스 전공 석사 졸업논문 개요](images/README.png)

## MLflow + Docker 실행

### 1. 로컬 Python 환경에 의존성 설치

MLflow server는 Docker 컨테이너가 아니라 로컬 Python에서 실행하므로, 먼저 `mlflow`가 포함된 의존성을 설치합니다.

```bash
python3 -m pip install -r requirements.txt
```

가상환경을 쓰는 경우에는 활성화한 뒤 같은 명령을 실행하면 됩니다.

### 2. MLflow 저장 폴더 생성

```bash
mkdir -p result/mlflow/artifacts
```

### 3. MLflow 서버 실행

프로젝트 루트에서 아래 명령으로 MLflow UI와 tracking server를 먼저 실행합니다.

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

브라우저에서 [http://127.0.0.1:5001](http://127.0.0.1:5001) 으로 접속하면 됩니다.

터미널에서 실행 중인 MLflow 서버를 끌 때는 해당 터미널 창에서 `Ctrl + C`를 누르면 됩니다.

백그라운드로 실행했거나 어떤 터미널에서 띄웠는지 모르겠으면 아래처럼 종료할 수 있습니다.

```bash
lsof -iTCP:5001 -sTCP:LISTEN -n -P
kill <PID>
```

### 4. Docker 이미지 빌드

docker 실행 후 

```bash
docker build -t thesis-model .
```

### 5. Docker 컨테이너에서 학습 실행

하이퍼파라미터를 바꿔 실행합니다.

설정 가능한 주요 환경변수는 아래와 같습니다.

- `EXPERIMENT_PREFIX`: MLflow experiment 이름 prefix. 예: `1.실험`. 기본값은 `model-weighting-comparison`
- `EXPERIMENT_NAME`: MLflow experiment 이름 전체를 직접 지정할 때 사용합니다. 값을 주면 `EXPERIMENT_PREFIX`보다 우선합니다.
- `DATA_NAMES`: 사용할 데이터셋 선택값. `all`, `사출성형기`, `살균기`, `소성가공`, `용해탱크`, 또는 쉼표 목록을 받을 수 있습니다. 기본값은 `사출성형기`
- `MODEL_NAMES`: 사용할 모델 선택값. `all`, `MLPClassifier`, `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `SVC`, `XGBClassifier`, 또는 쉼표 목록을 받을 수 있습니다. 기본값은 `MLPClassifier`
- `RUN_DATA_LOADER`: 데이터 로드 단계 실행 여부. 기본값은 `1`
- `RUN_EDA`: EDA 단계 실행 여부. 기본값은 `1`
- `RUN_ANALYSIS`: 통계 분석 단계 실행 여부. 기본값은 `1`
- `RUN_MODELING`: 모델링 단계 실행 여부. 기본값은 `1`
- `FEATURE_IMPORTANCE_F1_THRESHOLD`: feature importance 반복 실행에서 최소 F1 기준. 기본값은 `0.0`
- `FEATURE_IMPORTANCE_RUNS`: feature importance 반복 횟수. 기본값은 `30`
- `FEATURE_IMPORTANCE_TOP_K`: feature importance로 선택할 상위 feature 개수. 기본값은 `10`
- `IDX_OUTLIER_DIRECTIONAL_CORR_THRESHOLD`: idx 이상치 제거 상관계수 임계값. 기본값은 `0.3`
- `DATE_TREND_CONTEXT_DAYS`: high defect anchor 날짜 전후로 비교할 일수. 기본값은 `2`
- `DATE_TREND_MIN_WINDOW_POINTS`: anchor 주변 패턴 비교에 필요한 최소 날짜 수. 기본값은 `4`
- `DATE_TREND_MIN_ANCHOR_RATE_QUANTILE`: high defect anchor로 볼 defect rate 분위수 기준. 기본값은 `0.8`
- `DATE_TREND_MIN_LEVEL_SCORE`: anchor 날짜에서 feature가 충분히 높거나 낮다고 볼 z-score 기준. 기본값은 `0.5`
- `DATE_TREND_MIN_SIGN_AGREEMENT`: anchor 주변 구간에서 defect rate와 feature 증감 부호가 일치해야 하는 최소 비율. 기본값은 `0.5`
- `MLP_HIDDEN_LAYER_SIZES`: MLP 은닉층 크기. 쉼표로 구분하며 기본값은 `64,32`
- `MLP_ACTIVATION`: MLP 활성화 함수. 기본값은 `relu`
- `MLP_SOLVER`: MLP optimizer. 기본값은 `adam`
- `MLP_MAX_ITER`: MLP 최대 반복 수. 기본값은 `1000`
- `MLP_RANDOM_STATE`: MLP random seed. 기본값은 `42`
- `BASELINE_INDEX_MUL`: weighted 실행에 사용할 index 가중치 배수. 기본값은 `2`
- `BASELINE_DATE_MUL`: weighted 실행에 사용할 date 가중치 배수. 기본값은 `2`

예를 들어 아래처럼 `1.실험` prefix를 주고, 데이터와 모델을 선택해서 실행할 수 있습니다.

```bash
docker run --rm \
  -e EXPERIMENT_PREFIX="1.실험" \
  -e DATA_NAMES="살균기" \
  -e MODEL_NAMES="MLPClassifier" \
  -e RUN_DATA_LOADER=1 \
  -e RUN_EDA=0 \
  -e RUN_ANALYSIS=1 \
  -e RUN_MODELING=1 \
  -e FEATURE_IMPORTANCE_F1_THRESHOLD=0.0 \
  -e FEATURE_IMPORTANCE_RUNS=30 \
  -e FEATURE_IMPORTANCE_TOP_K=10 \
  -e IDX_OUTLIER_DIRECTIONAL_CORR_THRESHOLD=0.3 \
  -e DATE_TREND_CONTEXT_DAYS=2 \
  -e DATE_TREND_MIN_WINDOW_POINTS=4 \
  -e DATE_TREND_MIN_ANCHOR_RATE_QUANTILE=0.8 \
  -e DATE_TREND_MIN_LEVEL_SCORE=0.5 \
  -e DATE_TREND_MIN_SIGN_AGREEMENT=0.5 \
  -e MLP_HIDDEN_LAYER_SIZES=64,32 \
  -e MLP_ACTIVATION=relu \
  -e MLP_SOLVER=adam \
  -e MLP_MAX_ITER=1000 \
  -e MLP_RANDOM_STATE=42 \
  -e BASELINE_INDEX_MUL=2 \
  -e BASELINE_DATE_MUL=2 \
  -e MLFLOW_TRACKING_URI="http://host.docker.internal:5001" \
  --name thesis-model \
  thesis-model
```

전체 데이터셋과 전체 모델을 모두 돌리려면 아래처럼 실행하면 됩니다.

```bash
docker run --rm \
  -e EXPERIMENT_PREFIX="[260813_1]" \
  -e DATA_NAMES="all" \
  -e MODEL_NAMES="all" \
  -e RUN_DATA_LOADER=1 \
  -e RUN_EDA=0 \
  -e RUN_ANALYSIS=1 \
  -e RUN_MODELING=1 \
  -e MLFLOW_TRACKING_URI="http://host.docker.internal:5001" \
  --name thesis-model \
  thesis-model
```

여러 개만 골라서 돌릴 때는 쉼표로 넘기면 됩니다.

```bash
docker run --rm \
  -e EXPERIMENT_PREFIX="2.비교실험" \
  -e DATA_NAMES="사출성형기,살균기" \
  -e MODEL_NAMES="MLPClassifier,RandomForestClassifier,XGBClassifier" \
  -e RUN_DATA_LOADER=1 \
  -e RUN_EDA=0 \
  -e RUN_ANALYSIS=1 \
  -e RUN_MODELING=1 \
  -e MLFLOW_TRACKING_URI="http://host.docker.internal:5001" \
  --name thesis-model \
  thesis-model
```

컨테이너에서 실행된 결과는 앞에서 띄운 MLflow 서버로 기록되고, 실험 메타데이터와 artifact는 프로젝트의 `result/mlflow` 폴더에 저장됩니다.
Docker 컨테이너처럼 원격 클라이언트가 MLflow 서버에 기록할 때는 `--serve-artifacts`가 꼭 필요합니다. 이 옵션이 없으면 params, metrics, tags는 저장되어도 artifact는 서버를 통해 업로드되지 않아 UI에 `No Artifacts Recorded`로 보일 수 있습니다.

MLflow experiment 이름은 `EXPERIMENT_NAME`으로 직접 지정할 수 있고, 비워두면 `{EXPERIMENT_PREFIX}--{DATA_NAME}` 형식으로 자동 생성됩니다. 예를 들어 `EXPERIMENT_PREFIX="1.실험"`이고 `DATA_NAMES="살균기"`이면 experiment 이름은 `1.실험--살균기`가 됩니다.
여러 데이터셋을 선택하면 데이터셋별로 각각 experiment가 만들어집니다.
부모 run 이름은 현재 모델명으로 기록되고, child run 이름은 `{MODEL_NAME}-baseline`, `{MODEL_NAME}-weighted`로 기록됩니다.
MLflow metric에는 `test_accuracy`, `validation_accuracy`, `cv_accuracy`, `cv_f1score`, `cv_precision`, `cv_recall`가 기록됩니다.
`weighted` child run의 artifact에는 `weighting_details/` 아래에 가중치 적용 대상과 요약 CSV가 저장됩니다.

### 6. 결과 확인

MLflow UI에서 run, metric, artifact를 확인합니다.

### 참고

- macOS, Windows Docker Desktop 기준으로 컨테이너에서 호스트 MLflow 서버에 접속할 때 `host.docker.internal`을 사용합니다.
- MLflow 3.5 이상에서는 Host header 보안 검사가 있어서, Docker에서 접속하려면 `--allowed-hosts "127.0.0.1:5001,localhost:5001,host.docker.internal:5001"` 같은 설정이 필요합니다.
- Linux에서는 필요하면 `--add-host=host.docker.internal:host-gateway` 옵션을 추가한 뒤 같은 `MLFLOW_TRACKING_URI`를 사용하면 됩니다.
