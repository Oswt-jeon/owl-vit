# OWL-ViT 개발 컨테이너 운용 가이드

이 문서는 OWL-ViT 프로젝트의 Docker 컨테이너를 계속 실행 상태로 유지하고, 호스트의 소스 및 이미지를 컨테이너에서 바로 확인할 수 있도록 설정하는 방법을 정리합니다.

## 1. 이미지 빌드

```bash
docker build -t owlvit-dev .
```

## 2. 컨테이너 백그라운드 실행 (루트 디렉터리 마운트)

```bash
docker run -d --name owlvit-running --gpus all \
  -v "$(pwd)":/app -w /app \
  owlvit-dev sleep infinity
```

- `-v "$(pwd)":/app`: 현재 프로젝트 디렉터리를 컨테이너의 `/app` 에 마운트하여 코드·이미지 변경 사항을 즉시 반영합니다.
- `sleep infinity` 는 컨테이너가 종료되지 않고 계속 유지되도록 합니다. 필요하다면 `tail -f /dev/null` 등을 사용할 수도 있습니다.

## 3. 컨테이너 내부에서 스크립트 실행

컨테이너가 띄워진 상태에서 다음 명령으로 스크립트를 실행합니다.

```bash
docker exec -it owlvit-running python run_owlvit.py
```

필요할 때마다 새 터미널에서 동일한 명령을 반복 실행하면 됩니다.

## 4. 컨테이너 중지 및 정리

```bash
docker rm -f owlvit-running
```

다시 실행해야 할 경우 2단계의 `docker run ...` 명령을 재사용합니다.
