import torch
from torchvision import datasets, transforms
import random
import numpy as np
import ast
from tqdm import tqdm


mnist_dataset = datasets.MNIST(root="data/mnist_data/", transform = transforms.ToTensor(), train=True, download=True)
# chessboard = torch.tensor([[1, 2, 3, 4],
#                            [2, 3, 4, 1],
#                            [3, 4, 1, 2],
#                            [4, 1, 2, 3]])
# chessboard = torch.nn.functional.one_hot(chessboard - 1)
# label = torch.argmax(chessboard, dim=-1) + 1
# chessboard_img = torch.zeros((4, 4, 28, 28))

with open('data/sudoku4data/all_poss.txt', 'r') as f:
    all_sol_string = f.read()
with open('data/sudoku4data/unique_pos.txt', 'r') as f2:
    all_masks_string = f2.read()

all_sols = ast.literal_eval(all_sol_string)
all_sols = np.array(all_sols, dtype='int32')
all_masks = ast.literal_eval(all_masks_string)
sol_data = []
for pos, mask in all_masks:
    sol_data.append(all_sols[pos].reshape(4, 4))
sol_data = np.array(sol_data)
torch_sol = torch.from_numpy(sol_data)
torch_sol = torch.nn.functional.one_hot(torch_sol.flatten().long())
sols = torch_sol.view(85632, 4, 4, -1).float()
Y_in = sols[:, :, :, 1:]
Y_in = torch.argmax(Y_in, dim=-1) + 1
X_img = torch.zeros((85632, 4, 4, 28, 28))

for i in tqdm(range(85632)):
    for row in range(4):
        for col in range(4):
            digit_indices = (mnist_dataset.targets == Y_in[i, row, col]).nonzero(as_tuple=False).squeeze()
            selected_index = random.choice(digit_indices)
            selected_image, _ = mnist_dataset[selected_index]
            X_img[i, row, col] = selected_image.squeeze(0) * 255.
torch.save(X_img, 'data/sudoku4data/features_img.pt')
