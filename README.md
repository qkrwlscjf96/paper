## 논문 개요
![서강대학교 데이터 사이언스 전공 석사 졸업논문 개요](images/README.png)

## Docker 실행

이미지 빌드:

```bash
docker build -t thesis-model .
```

조합 (DATA_NAME / INDEX_MULS / DATE_MULS)을 바꾸고 파일경로 바꿔 실행:

```bash
docker run --rm \
  -e DATA_NAME=사출성형기 \
  -e INDEX_MULS=1,2,3 \
  -e DATE_MULS=1,2,3 \
  -e MLFLOW_TRACKING_URI="sqlite:////Users/danielpark/Documents/서강대/pgm/논문/mlflow_data/mlflow.db" \
  -e MLFLOW_ARTIFACT_ROOT="file:///Users/danielpark/Documents/서강대/pgm/논문/mlflow_data/artifacts" \
  -v "$(pwd)/mlflow_data:/Users/danielpark/Documents/서강대/pgm/논문/mlflow_data" \
  --name thesis-model \
  thesis-model
```

실험 결과는 프로젝트의 `mlflow_data` 폴더에 저장됩니다.

MLflow UI 실행:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python3 -m mlflow server \
  --backend-store-uri "sqlite:////Users/danielpark/Documents/서강대/pgm/논문/mlflow_data/mlflow.db" \
  --default-artifact-root "file:///Users/danielpark/Documents/서강대/pgm/논문/mlflow_data/artifacts" \
  --host 127.0.0.1 \
  --port 5000
```
