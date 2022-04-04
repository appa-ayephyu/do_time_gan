def soft_update(online, target, tau=0.9):
    for online, target in zip(online.parameters(), target.parameters()):
        target.data = tau * target.data + (1 - tau) * online.data
