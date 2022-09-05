"""
예제 출처
https://tutorials.pytorch.kr/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py
"""

import torch
import numpy as np

# 데이터로 부터 직접 tensor 생성
print("데이터로 부터 직접 tensor 생성")
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print(x_data)

# Numpy 배열로 생성
print("Numpy 배열로 생성")
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# 다른 tensor로 부터 생성하기
print("다른 tensor로 부터 생성하기")
x_ones = torch.ones_like(x_data)  # x_data의 속성을 유지합니다.
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # x_data의 속성을 덮어씁니다.
print(f"Random Tensor: \n {x_rand} \n")

# tensor 속성
print("tensor 속성")
tensor = torch.rand(3, 4)  # 4*3로 랜덤으로 생성
print(tensor)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# tensor 연산
# GPU가 존재하면 텐서를 이동합니다. 없는 경우 그냥 pass됨
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on: {tensor.device}")

t1 = torch.cat([tensor, tensor, tensor], dim=1)  # tensor 합치기 사실상 일렬로 연결 같은 row끼리 연결
print(t1)

# 요소별 곱(element-wise product)을 계산합니다
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# 다른 방법
print(f"tensor * tensor \n {tensor * tensor}")

# 행렬 곱
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# 다른 문법:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# replace 연산
print("replace 연산")
print(tensor, "\n")
tensor.add_(5)  # 모든 값에 5씩 더함
print(tensor)

# tensor -> Numpy로 변경
print("tensor -> Numpy로 변경")
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# tensor 변경사항이 Numpy에도 적용
print("tensor 변경사항이 Numpy에도 적용")
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# Numpy -> tensor로 변경
print("Numpy -> tensor로 변경")
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
