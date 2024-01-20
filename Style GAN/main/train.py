from tqdm import tqdm
from constants.variable import *
from utilities.utils import *
from main.GAN import * 
import wandb 
wandb.login()



def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
):
    loop = tqdm(loader, leave=True)
    wandb.init(
        project="Style Gan session 1")
    
    # Copy your config 
    name_proj = wandb.config

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]

        noise = torch.randn(cur_batch_size, Z_DIM).to(DEVICE)

        fake = gen(noise, alpha, step)
        critic_real = critic(real, alpha, step)
        critic_fake = critic(fake.detach(), alpha, step)
        gp = gradient_penalty(critic, real, fake, alpha, step, device=DEVICE)
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + LAMBDA_GP * gp
            + (0.001 * torch.mean(critic_real**2))
        )
        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / ((PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

        # Log metrics to wandb
        wandb.log({
            'critic_real':critic_real,
            'critic_fake':critic_fake,
            'gen_fake':gen_fake,
            'alpha': alpha,
            'gp': gp.item(),
            'loss_critic': loss_critic.item()
        })

    return alpha

gen = Generator(
        Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG
    ).to(DEVICE)
critic = Discriminator(IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
# initialize optimizers
opt_gen = optim.Adam([{"params": [param for name, param in gen.named_parameters() if "map" not in name]},
                        {"params": gen.map.parameters(), "lr": 1e-5}], lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_critic = optim.Adam(
    critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99)
)

gen.train()
critic.train()



# start at step that corresponds to img size that we set in config
step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))
for num_epochs in PROGRESSIVE_EPOCHS[step:]:
    alpha = 1e-5   # start with very low alpha
    loader, dataset = get_loader(4 * 2 ** step)  
    print(f"Current image size: {4 * 2 ** step}")

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        alpha = train_fn(
            critic,
            gen,
            loader,
            dataset,
            step,
            alpha,
            opt_critic,
            opt_gen
        )

    generate_examples(gen, step)
    step += 1  # progress to the next img size


# Save the generator and critic models in .pth format
torch.save(gen.state_dict(), 'generator.pth')
torch.save(critic.state_dict(), 'critic.pth')

# Save the generator and critic models in compressed format
torch.save(gen.state_dict(), 'generator_compressed.pth.tar')
torch.save(critic.state_dict(), 'critic_compressed.pth.tar')
