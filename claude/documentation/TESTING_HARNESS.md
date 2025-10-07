name: ARM64 Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-arm64:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up QEMU
      uses: docker/setup-qemu-action@v2
      with:
        platforms: arm64

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Build ARM64 test image
      run: |
        docker buildx build \
          --platform linux/arm64 \
          -f Dockerfile.rpi \
          -t wildlife-camera:arm64-test \
          --load \
          .

    - name: Run unit tests
      run: |
        docker run --platform linux/arm64 \
          -v ${{ github.workspace }}:/app \
          wildlife-camera:arm64-test \
          python3 -m pytest tests/test_optical_flow_analyzer.py -v

    - name: Run integration tests
      run: |
        docker run --platform linux/arm64 \
          -v ${{ github.workspace }}:/app \
          wildlife-camera:arm64-test \
          python3 test_integration.py

    - name: Run import tests
      run: |
        docker run --platform linux/arm64 \
          -v ${{ github.workspace }}:/app \
          wildlife-camera:arm64-test \
          python3 test_imports.py

    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: test-results-arm64
        path: test_results/
