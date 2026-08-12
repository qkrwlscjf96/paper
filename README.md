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
mkdir -p mlflow_data/artifacts
```

### 3. MLflow 서버 실행

프로젝트 루트에서 아래 명령으로 MLflow UI와 tracking server를 먼저 실행합니다.

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python3 -m mlflow server \
  --backend-store-uri "sqlite:///$PWD/mlflow_data/mlflow.db" \
  --default-artifact-root "file://$PWD/mlflow_data/artifacts" \
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

### 4. Docker 이미지 빌


docker 실행 후 

```bash
docker build -t thesis-model .
```

### 5. Docker 컨테이너에서 학습 실행

`DATA_NAME`, `INDEX_MULS`, `DATE_MULS` 조합을 바꿔 실행합니다.

```bash
docker run --rm \
  -e DATA_NAME=사출성형기 \
  -e INDEX_MULS=1,2,3 \
  -e DATE_MULS=1,2,3 \
  -e MLFLOW_TRACKING_URI="http://host.docker.internal:5001" \
  --name thesis-model \
  thesis-model
```

컨테이너에서 실행된 결과는 앞에서 띄운 MLflow 서버로 기록되고, 실험 메타데이터와 artifact는 프로젝트의 `mlflow_data` 폴더에 저장됩니다.

### 6. 결과 확인

MLflow UI에서 run, metric, artifact를 확인합니다.

### 참고

- macOS, Windows Docker Desktop 기준으로 컨테이너에서 호스트 MLflow 서버에 접속할 때 `host.docker.internal`을 사용합니다.
- MLflow 3.5 이상에서는 Host header 보안 검사가 있어서, Docker에서 접속하려면 `--allowed-hosts "127.0.0.1:5001,localhost:5001,host.docker.internal:5001"` 같은 설정이 필요합니다.
- Linux에서는 필요하면 `--add-host=host.docker.internal:host-gateway` 옵션을 추가한 뒤 같은 `MLFLOW_TRACKING_URI`를 사용하면 됩니다.
