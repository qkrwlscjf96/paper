# 베이스 이미지
FROM python:3.11.5

# 작업 디렉토리 생성
WORKDIR /app

# requirements 복사
COPY requirements.txt .

# 패키지 설치
RUN pip install -r requirements.txt

# 코드 복사
COPY . .

# 실행 명령
CMD ["python", "-m", "src.main"]