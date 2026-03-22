# 베이스 이미지
FROM python:3.11.5-slim

# 컨테이너 런타임 환경
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# xgboost / scipy 계열 런타임 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 생성
WORKDIR /app

# requirements 복사
COPY requirements.txt .

# 패키지 설치
RUN pip install --upgrade pip && pip install -r requirements.txt

# 코드 복사
COPY . .

# 캐시 디렉토리 생성
RUN mkdir -p /tmp/matplotlib /tmp/.cache

# 실행 명령
CMD ["python", "-u", "-m", "src.main"]
