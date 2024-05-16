import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.tensor([1.0], device=device)
y = torch.tensor([0.0], device=device)
w = torch.rand(1, requires_grad=True, device=device)
b = torch.rand(1, requires_grad=True, device=device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD([w, b], lr=1e-3)

for i in range(10000):
    z = w * x + b
    loss = loss_fn(z, y)
    loss.backward()

    print("w.grad:", w.grad)
    print("b.grad:", b.grad)

    optimizer.step()
    optimizer.zero_grad()

    print("Updated w:", w)
    print("Updated b:", b)

    print("loss:", loss)
    print("z:", z, "result:", y)
