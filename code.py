import torch
import gc

# 현재 캐시된 메모리 정리
torch.cuda.empty_cache()

# 할당된 텐서 객체도 정리 (필수 아님, 메모리 누수 방지용)
gc.collect()

# 사용 예: 에러 발생 후 또는 대규모 모델 삭제 후
