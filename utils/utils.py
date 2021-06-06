def weights_init(m):
    """
    initializaing weights
    """
    initrange = 0.1
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Embedding') != -1:
        m.weight.data.normal_(0, 0.01)


def get_ave_vel(trajs):
    ave_vel_list = []
    for traj in trajs:
        ave_vel = 0
        for t in range(1, len(traj)):
            cur_vel = ((traj[t][0] - traj[t - 1][0]) ** 2 + (traj[t][0] - traj[t - 1][0]) ** 2) ** 0.5
            ave_vel += cur_vel
        if len(traj) > 1:
            ave_vel /= len(traj) - 1
        ave_vel_list.append(ave_vel)
    return ave_vel_list
