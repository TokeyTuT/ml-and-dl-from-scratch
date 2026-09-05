import time
import torch
from datasets import load_dataset
from EncoderDecoder import Transformer
from processing import load_multi30k






def train(
        net,
        train_iter,
        test_iter,
        num_epochs,
        loss,
        device
):
    
    print(f'training on {device}...')
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        net.train()
        # 修改：按有效目标 token 数累计整轮训练损失。
        train_total_l, train_n = 0.0, 0
        for batch in train_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            Y_hat = net(X, Y[:, :-1], X_valid_len)
            l = loss(Y_hat.reshape(-1, Y_hat.shape[-1]), Y[:, 1:].reshape(-1))
            l.backward()
            optimizer.step()
            num_tokens = (Y[:, 1:] != loss.ignore_index).sum().item()
            train_total_l += l.item() * num_tokens
            train_n += num_tokens
        print(f'epoch {epoch + 1}, loss {train_total_l / train_n:f}')

        print('evaluating on validation set...')
        net.eval()
        with torch.no_grad():
            total_l, n = 0.0, 0
            for batch in test_iter:
                X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
                Y_hat = net(X, Y[:, :-1], X_valid_len)
                l = loss(Y_hat.reshape(-1, Y_hat.shape[-1]), Y[:, 1:].reshape(-1))
                # 修改：CrossEntropyLoss 默认按非 padding token 平均。
                num_tokens = (Y[:, 1:] != loss.ignore_index).sum().item()
                total_l += l.item() * num_tokens
                n += num_tokens
            print(f'validation loss {total_l / n:f}')
        
        
        
    
def main():

    batch_size,num_steps,min_freq,reserved_tokens = 32,32,2,None
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')


    begin_time = time.time()

    # 修改：第五个参数是 dataset，设备迁移已在训练循环中完成。
    loaders,src_vocab,tgt_vocab = load_multi30k(batch_size,num_steps,min_freq,reserved_tokens)

    src_vocab = src_vocab
    tgt_vocab = tgt_vocab

    net = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab), # type: ignore
        num_layers=2,
        num_heads=4,
        d_model=128,
        ffn_num_hiddens=512,
        dropout=0.1,
        use_bias=False
    )

    loss = torch.nn.CrossEntropyLoss(ignore_index=tgt_vocab['<padding>']) # type: ignore

    train(
        net=net,
        train_iter=loaders['train'],
        test_iter=loaders['validation'],
        num_epochs=10,
        loss=loss,
        device=device
    )


if __name__ =='__main__':
    # 修改：执行 demo 入口。
    main()
