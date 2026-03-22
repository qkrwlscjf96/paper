![서강대학교 데이터 사이언스 전공 석사 졸업논문 개요](images/README.png)

## Docker 실행

이미지 빌드:

```bash
docker build -t thesis-model .
```

컨테이너 실행:

```bash
docker run --rm --name thesis-model thesis-model
```

직접 셸로 들어가 확인:

```bash
docker run --rm -it --name thesis-model-shell thesis-model /bin/bash
```
