import torch

# torch.hub.list("zhanghang1989/ResNeSt", force_reload=True)
# # load pretrained models, using ResNeSt-50 as an example
# model = torch.hub.load("zhanghang1989/ResNeSt", "resnest101", pretrained=False)

# state_dict = torch.load("/home/ubuntu/final-project-challenge-3-so_ez_peasy/william/best.pt", map_location='cuda:2')

# model.load_state_dict(state_dict["model_state_dict"])

# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# optimizer.load_state_dict(state_dict["optimizer_state_dict"])

# optimizer2 = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20])
# scheduler.load_state_dict(state_dict["scheduler_state_dict"])

# optimizer.param_groups[0].update({'momentum':0.9})

# # print(optimizer.state_dict()['param_groups'][0]['momentum'])
# print(scheduler.state_dict())

from lib.scheduler import CustomScheduler

conv = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.Linear(10, 2))
opt = torch.optim.SGD(conv.parameters(), lr=0.001)
sch = CustomScheduler(opt, {0: 0.001, 10:0.0001, 20:0.01})
input = torch.randn(10, 10)
target = torch.randint(low=0, high=2, size=(10, ))
criterion = torch.nn.CrossEntropyLoss()
for e in range(20):
    logits = conv(input)
    loss = criterion(logits, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    sch.step()
    print(f'{e}: {opt.param_groups[0]["lr"]}')
