from torch.utils.data import Dataset, DataLoader

class SampleDataset(Dataset):
    def __init__(self):
        self.data = [1, 2, 3, 4, 5]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = SampleDataset()
loader = DataLoader(dataset, batch_size=2)

for batch in loader:
    print(batch)