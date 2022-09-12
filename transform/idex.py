import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# 데이터가 항상 머신러닝 알고리즘 학습에 필요한 최종 처리가 된 형태로 제공되지는 않습니다.
# 변형(transform) 을 해서 데이터를 조작하고 학습에 적합하게 만듭니다.
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
print(ds)

# Lambda 변형은 사용자 정의 람다(lambda) 함수를 적용합니다.
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

print(target_transform)
