import torch
import torch_geometric

graph = torch_geometric.datasets.Planetoid(root='../', name='cora')[0]

f = open("net.txt","w")
for i in range(graph.edge_index.shape[1]):
    f.write(str(graph.edge_index[0,i].item())+"\t"+str(graph.edge_index[1,i].item())+"\t"+"1"+"\n")

f = open("label.txt","w")
for i in range(graph.x.shape[0]):
    f.write(str(i)+"\t"+str(graph.y[i].item())+"\n")

f = open("train.txt", "w")
for i in range(graph.x.shape[0]):
    if graph.train_mask[i].item():
        f.write(str(i)+"\n")

f = open("dev.txt", "w")
for i in range(graph.x.shape[0]):
    if graph.val_mask[i].item():
        f.write(str(i)+"\n")

f = open("test.txt", "w")
for i in range(graph.x.shape[0]):
    if graph.test_mask[i].item():
        f.write(str(i)+"\n")

f = open("feature.txt", "w")
for i in range(graph.x.shape[0]):
    f.write(str(i))
    count = 0
    for j in range(graph.x.shape[1]):
        if graph.x[i,j].item():
            if not count:
                f.write("\t"+str(j)+":"+str(graph.x[i,j].item()))
                count += 1
            else:
                f.write(" "+str(j)+":"+str(graph.x[i,j].item()))
    else:
        f.write("\n")
