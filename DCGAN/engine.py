import torch
import matplotlib.pyplot as plt
from utils import checkpoint


def run(gen_net,
        dis_net,
        dataloader,
        device,
        loss_fn,
        optimizer_gen,
        optimizer_dis,
        Exp_ID,
        epochs,
        config,
        root
        ):

    gen_net.train()
    dis_net.train()

    loss_generator, loss_discriminator = [], []
    for epoch in range(epochs):

        print(f"_________________ Epoch: {epoch + 1} _________________")

        loss_g, loss_d = 0, 0
        for batch_count, (img, _) in enumerate(dataloader):

            img = img.to(device)

            # G E N E R A T O R

            z = torch.randn(size=(img.shape[0], config["z_dim"])).to(device)

            fake_img = gen_net(z)

            if epoch % 10 == 0 and batch_count in [0, 100]:
                plt.figure()
                plt.imshow(fake_img[0].cpu().detach().permute(2, 1, 0))
                plt.imsave()
                plt.show()
                plt.close()

            gen_label = torch.ones(size=(img.shape[0],)).unsqueeze(dim=1).to(device)
            gen_loss = loss_fn(dis_net(fake_img), gen_label)

            loss_g += gen_loss

            optimizer_gen.zero_grad()
            gen_loss.backward()
            optimizer_gen.step()

            # D I S C R I M I N A T O R (We May Replace Dis_Net With ViT)

            # IMPORTANT NOTE:
            # We have to use detach fake_images to make sure that while updating the discriminator net,
            # the gradients does not reach the generator network
            fake_class_pred = dis_net(fake_img.detach())
            real_class_pred = dis_net(img.type(torch.float))

            fake_class_label = torch.zeros(size=(img.shape[0],)).unsqueeze(dim=1).to(device)
            real_class_label = torch.ones(size=(img.shape[0],)).unsqueeze(dim=1).to(device)

            fake_loss = loss_fn(fake_class_pred, fake_class_label)
            real_loss = loss_fn(real_class_pred, real_class_label)

            dis_loss = fake_loss + real_loss

            loss_d += dis_loss

            optimizer_dis.zero_grad()
            dis_loss.backward()
            optimizer_dis.step()
            # print("reached")
            # print(f"{batch_count}")

        loss_g /= len(dataloader)
        loss_d /= len(dataloader)

        loss_generator.append(loss_g.item())
        loss_discriminator.append(loss_d.item())

        print(f"Loss G: {loss_g} \nLoss D: {loss_d}")

        checkpoint(loss_generator=loss_generator,
                   loss_discriminator=loss_discriminator,
                   root=root,
                   Exp_ID=Exp_ID,
                   epoch=epoch,
                   config=config)







