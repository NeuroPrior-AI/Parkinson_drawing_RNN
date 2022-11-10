
from .config import *
from .model import *
lstm = Net(num_classes, input_size, hidden_size, num_layers, 200).to(device)
criterion = torch.nn.BCELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
def train_epoch(model, train_loader, test_loader, num_epoch=num_epochs, criterion=criterion, optimizer=optimizer):
    model.train()
    for i in range(num_epochs):
        for seq, labels in train_loader:
            model = model.to(device)
            seq = seq.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            y_pred = model(seq)

            y_pred = y_pred.to(torch.float32)
            labels = labels.to(torch.float32)

            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()

        if i % 100 == 0:
            # print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
            acc = eval_performance(model, test_loader)
            print("Epoch: %d, loss: %1.5f, acc:%1.5f" % (i, loss.item(), acc))


def eval_performance(model, test_loader):
    total_correct = 0
    total_num = 0

    model.eval()
    with torch.no_grad():
        for seq, labels in test_loader:
            seq = seq.to(device)
            labels = labels.to(device)

            y_pred = model(seq)

            y_pred = y_pred > 0.5
            total_correct += torch.eq(y_pred, labels).float().sum().item()
            total_num += seq.size(dim=0)
        acc = total_correct / total_num

    model.train()
    return acc
